#!/usr/bin/env bash
# capture_run.sh — 실차 주행 1회를 "분석 가능한 상태로" 통째 저장.
#
# 백(rosbag)만 뜨면 나중에 분석이 막힙니다. 파라미터·설정·레이스라인·코드 버전은
# 토픽으로 흐르지 않기 때문입니다. 이 스크립트는 그것들을 백과 같은 폴더에
# 스냅샷으로 남긴 뒤 녹화를 시작합니다.
#
#   사용법 (차량, race.launch 가 이미 떠 있는 상태에서):
#       ./capture_run.sh <라벨>
#   예:
#       ./capture_run.sh v9_map          # v9 파라미터 주행
#       ./capture_run.sh baseline_map    # 비교용 기존 파라미터 주행
#
#   Ctrl-C 로 녹화를 끝냅니다. 결과: ~/runs/<날짜시각>_<라벨>/
#
# 주의: 같은 세션에서 두 설정을 비교하려면 배터리·타이어·노면이 같아야 하므로
#       연속으로 찍고, 라벨만 다르게 주세요.
set -u

LABEL="${1:-run}"
OUT=~/runs/$(date +%m%d_%H%M)_"$LABEL"
STACK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT"/meta

echo "[capture] -> $OUT"

# ── 1. 코드 버전 ──────────────────────────────────────────────────────
{
    echo "label:        $LABEL"
    echo "recorded_at:  $(date -Is)"
    echo "host:         $(hostname)"
    echo "stack_path:   $STACK"
    echo "git_commit:   $(git -C "$STACK" rev-parse HEAD 2>/dev/null || echo 'n/a')"
    echo "git_dirty:    $(git -C "$STACK" status --porcelain 2>/dev/null | wc -l) files"
} > "$OUT"/meta/run_info.yaml
git -C "$STACK" status --porcelain > "$OUT"/meta/git_dirty.txt 2>/dev/null

# ── 2. 실행 중인 launch 명령 (map / use_map / lu_table / localization) ──
ps -eo args= | grep '[r]os2 launch' > "$OUT"/meta/launch_cmdline.txt 2>/dev/null
ros2 node list      > "$OUT"/meta/node_list.txt 2>&1
ros2 topic list     > "$OUT"/meta/topic_list.txt 2>&1

# 어떤 로컬라이제이션이 도는지: 노드 이름으로 판정 (측정오차 분석에 필수)
{
    grep -qi cartographer "$OUT"/meta/node_list.txt && echo "localization: cartographer"
    grep -qi particle     "$OUT"/meta/node_list.txt && echo "localization: pf"
    grep -qi kiss         "$OUT"/meta/node_list.txt && echo "localization: kiss"
} >> "$OUT"/meta/run_info.yaml

# ── 3. 살아있는 파라미터 (백에는 안 담김) ───────────────────────────────
for N in /controller_manager /state_machine; do
    ros2 param dump "$N" > "$OUT"/meta/params$(echo "$N" | tr / _).yaml 2>&1 || true
done

# measure 확인. /controller/debug 퍼블리셔는 노드 __init__ 에서만 만들어지므로
# 나중에 param set 으로 켤 수 없습니다. 꺼진 채로 녹화하면 그 토픽이 조용히
# 빈 채로 남고, 주행이 끝난 뒤에야 알게 됩니다 — 지금 멈추는 편이 낫습니다.
MEASURE=$(ros2 param get /controller_manager measure 2>/dev/null | grep -oiE 'true|false' | head -1)
echo "measure:      ${MEASURE:-unknown}" >> "$OUT"/meta/run_info.yaml
if [ "${MEASURE,,}" != "true" ]; then
    echo
    echo "  !! measure=${MEASURE:-unknown} 입니다. /controller/debug 가 발행되지 않습니다."
    echo "     퍼블리셔는 노드 시작 시점에만 생성되므로 'ros2 param set' 으로는 못 켭니다."
    echo "     스택을 measure:=true 로 다시 띄우세요:"
    echo
    echo "       ros2 launch stack_master race.launch.xml sim:=false map:=$(grep -oP 'map:=\K\S+' "$OUT"/meta/launch_cmdline.txt | head -1 || echo s) use_map:=true measure:=true"
    echo
    read -r -p "  그래도 이대로 녹화할까요? [y/N] " ANS
    [ "${ANS,,}" = "y" ] || { echo "  중단합니다."; rm -rf "$OUT"; exit 1; }
fi

# ── 4. 설정 스냅샷 ────────────────────────────────────────────────────
MAP=$(grep -oP 'map:=\K\S+' "$OUT"/meta/launch_cmdline.txt | head -1)
MAP="${MAP:-s}"
echo "map:          $MAP" >> "$OUT"/meta/run_info.yaml
for F in "$STACK/stack_master/config/controller.yaml" \
         "$STACK/stack_master/config/controller_map.yaml" \
         "$STACK/stack_master/config/CAR/vehicle_config.yaml" \
         "$STACK/stack_master/config/CAR/dynamics.yaml" \
         "$STACK/stack_master/maps/$MAP/speed_scaling.yaml" \
         "$STACK/stack_master/maps/$MAP/global_waypoints.json"; do
    [ -f "$F" ] && cp "$F" "$OUT"/meta/ 2>/dev/null
done
# 레이스라인이 시뮬과 같은 파일인지 나중에 대조할 수 있게 해시
md5sum "$OUT"/meta/global_waypoints.json 2>/dev/null >> "$OUT"/meta/run_info.yaml

# ── 5. 사람이 적는 메모 (자동으로는 알 수 없는 것) ──────────────────────
cat > "$OUT"/meta/NOTES.md <<'EOF'
# 주행 메모 (직접 채워주세요)

- 타이어: (새것 / 사용됨 / 교체 후 몇 랩)
- 노면:   (평소와 같음 / 먼지 / 습기 / 다른 장소)
- 배터리: (완충 / 몇 번째 주행)   ← 전압은 /vesc/sensors/core 에 기록됨
- 주행 중 파라미터를 만졌나: (아니오 / 예 → 무엇을)
- 이상 현상: (진동한 구간, 벽 접촉, 이탈 등 — s 좌표나 코너 이름으로)
- 비교 대상: (이 주행과 짝지어 비교할 다른 런의 라벨)
EOF

# ── 6. 녹화 ───────────────────────────────────────────────────────────
echo "[capture] 메타 저장 완료. 녹화 시작 — 끝내려면 Ctrl-C"
ros2 bag record -o "$OUT"/bag \
  /scan /scan_raw /pf/viz/inferred_pose /map \
  /car_state/odom /car_state/odom_frenet \
  /local_waypoints /global_waypoints /behavior_strategy /state_machine \
  /vesc/high_level/ackermann_cmd /vesc/ackermann_cmd \
  /vesc/commands/servo/position /vesc/sensors/imu /vesc/sensors/core /vesc/odom \
  /imu/data /l1_distance /lookahead_point \
  /controller/latency /controller/debug /state_machine/latency /tf /tf_static

echo
echo "[capture] 완료 -> $OUT"
echo "[capture] meta/NOTES.md 를 지금 채워두세요 (나중에 기억 안 납니다)"
du -sh "$OUT"

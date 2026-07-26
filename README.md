# UNICORN Racing Stack — ROS 2 Jazzy

ROS 2 Jazzy 기반 **F1TENTH 자율주행 레이싱 스택**입니다. 인지 → 추적 → 예측 →
계획 → 상태머신 → 제어까지 전체 파이프라인과, SIL(software-in-the-loop) 테스트용
**f1tenth_gym** 시뮬레이터를 리포 안에 함께 담고 있습니다. 클론 → 빌드 → 실행이면
끝나는 자기완결형 구조입니다.

> ROS 1 (catkin) 스택은 업스트림 리포의 **`ros1`** 브랜치에 동결 보관돼 있습니다
> ([Credits](#credits) 참고).

RoboStack(conda)을 쓰기 때문에 OS·아키텍처에 구애받지 않습니다. 검증 플랫폼:

|  | Ubuntu x86_64 | Ubuntu arm64 | macOS arm64 | Windows |
|---|:---:|:---:|:---:|:---:|
| **상태** | ✅ 검증됨 | ✅ 검증됨 | 🔺 부분 지원 | ⬜ 미검증 |
| **하드웨어** | NUC, 데스크톱 | Jetson (Orin) | Mac mini, MacBook | conda |

**설치 경로 안내:** 아래의 **conda (RoboStack)** 경로만 테스트·지원됩니다.
**시스템 ROS 2 Jazzy (apt/rosdep)** 와 **Docker** 는 계획 단계이며 아직 공식
지원이 아닙니다.

## 시작하기

**RoboStack(conda)이 기본이자 검증된 경로입니다** — ROS 2 Jazzy와 모든 의존성을
하나의 conda 환경(`unicorn`)에 넣으며, 시스템 ROS를 건드리지 않습니다
(Linux·macOS). 아래 세 블록을 위에서 아래로 복사해 붙여넣으세요.

### 1. conda — 이미 있으면 건너뛰기

`conda`(또는 `mamba`)가 이미 PATH에 있으면 **이 단계를 건너뛰세요**. 없으면
**Miniforge** 설치를 권장합니다 (최소 구성, `conda-forge` 기본, 라이선스 조건 없음):

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh" && exec $SHELL
conda config --set auto_activate_base false
```

### 2. 클론

```bash
mkdir -p ~/unicorn_ws/src && cd ~/unicorn_ws/src
git clone https://github.com/ajm11223/racing_stack.git unicorn-racing-stack
cd unicorn-racing-stack
```

<details><summary>워크스페이스 구조</summary>

colcon **워크스페이스 루트**는 `~/unicorn_ws` (즉 `src/`를 담고 있는 디렉토리)이고,
이 리포는 `~/unicorn_ws/src/unicorn-racing-stack`에 위치합니다. 모든 의존 컴포넌트가
벤더링돼 있어 서브모듈이 없으며, 평범한 `git clone` 하나로 충분합니다.
</details>

### 3. 설치

스크립트 하나가 전부 처리합니다 — `unicorn` 환경 생성, `~/.bashrc`·`~/.zshrc`에
`unicorn` 별칭 등록, pip 레이어 설치, `quadprog` 교체, CycloneDDS용 OS 소켓 버퍼
확대, 그리고 빌드(Release)까지. bash·zsh 모두에서 동작합니다:

```bash
cd ~/unicorn_ws/src/unicorn-racing-stack
./setup_conda_onLaptop.sh   # 시뮬/노트북: 하드웨어 전용 노드는 제외
# ./setup_conda_onCar.sh    # 차량: 전체 빌드
```

노트북 빌드는 **하드웨어 전용** 패키지(`urg_node`, `vesc_driver`/`vesc_ackermann`,
`particle_filter`)만 제외하고 나머지는 차량 스크립트를 그대로 실행합니다.
`vesc_msgs`는 **유지**되므로 VESC 메시지를 `ros2 topic echo`로 볼 수 있고,
RViz에서 서브맵을 보기 위한 `cartographer_rviz` 포함 cartographer 스택도 그대로입니다.
`setup_conda_onCar.sh`를 직접 실행하면 그 제외 목록을 지우고 전부 빌드합니다.

<details><summary>직접 단계별로 하고 싶다면 (같은 내용, 순서대로)</summary>

```bash
cd ~/unicorn_ws/src/unicorn-racing-stack
conda env create -f environment.yml                                              # conda 레이어: ROS 2 Jazzy + 의존성
echo "alias unicorn='source $(pwd)/unicorn.sh'" >> ~/.bashrc                      # 별칭 (zsh면 ~/.zshrc에도)
source unicorn.sh                                                                 # 지금 환경 진입
pip install -r requirements.txt                                                   # pip 레이어
pip install -e ./race_utils/unicorn_gym/f1tenth_gym                               # gym 코어 -> f110_gym
pip install --no-build-isolation -e ./race_utils/raycaster/range_libc/pywrapper   # range_libc
pip uninstall -y quadprog && conda install -y -c conda-forge quadprog=0.1.13      # quadprog 교체 (반드시 마지막)
cbuild                                                                            # colcon build (Release)
```
</details>

## 시뮬레이션 빠른 실행

설치 스크립트를 돌린 뒤 **새 셸을 열고**(또는 `source ~/.bashrc` / `~/.zshrc`):

```bash
unicorn   # 환경 진입: conda + PYTHONNOUSERSITE=1 + CycloneDDS + 워크스페이스 전부 소싱
ros2 launch stack_master race.launch.xml sim:=true map:=f   # 전체 자율주행 + 가상 상대차
#   low_level.launch.xml = 차량 + 센서만
```

**그다음 `a`를 눌러 자율주행을 arm 하세요** — 누르기 전까지 차는 멈춰 있습니다.
컨트롤러는 퍼블리시하지만 `keyboard_joy`가 시작 시 *human* 모드를 내보내는 동안
`simple_mux`가 아무것도 통과시키지 않기 때문입니다. 키 훅은 전역입니다
(pynput — 어떤 창이 포커스든 반응하므로, 평범하게 타이핑하다 걸릴 수 있습니다):

| 키 | 효과 |
|---|---|
| `a` | **auto** — 컨트롤러가 주행 |
| `h` | **human** — 수동 모드 (조작하지 않으면 정지) |
| 방향키 / 스페이스 | 수동 주행 / 명령 0 (human 모드) |

즉 정상 실행 후 차가 안 움직이면 대개 human 모드입니다. 더 깊이 파기 전에
`ros2 topic echo /joy_keyboard --once`로 확인하세요 (`buttons[5]=1`이 auto).
주행 중 아무 창에서나 `h`가 눌리면 차가 disarm 됩니다.

`unicorn`은 헬퍼도 정의합니다: `cbuild [패키지...]` (colcon build Release + 재소싱,
인자 없으면 워크스페이스 전체), `ros2kill` (모든 ROS 2 노드·런처·데몬 종료).

<details><summary><code>unicorn.sh</code>가 하는 일 — 그리고 항상 이걸로 진입해야 하는 이유</summary>

`conda activate`만 하지 말고 항상 `unicorn`(= `unicorn.sh` **소싱**)으로 진입하세요.
bash·zsh 모두에서 동작하며 다음을 수행합니다:
- **`PYTHONNOUSERSITE=1`** 설정 — 오래된 `~/.local/lib/python*`이 환경을 가리지 못하게;
- **CycloneDDS 선택** — `RMW_IMPLEMENTATION`은 `conda activate` *이후에* 설정해야
  합니다(activate가 이 변수를 지웁니다). 기본 FastDDS는 이 정도 규모의 노드 그래프에서
  코어를 busy-spin 하며 ~22 Hz에 그치는 반면, CycloneDDS는 ~21% CPU로 ~80 Hz를 냅니다.
  `CYCLONEDDS_URI`는 리포의 `cyclonedds.xml`을 가리킵니다(루프백 + Wi-Fi 인터페이스,
  멀티캐스트 on). `ROS_DOMAIN_ID`는 필요에 따라 조정하세요(기본 `1`);
- **ROS 환경을 깨끗한 기준으로 리셋** — rc 파일에서 소싱된 시스템 ROS나 다른
  워크스페이스가 conda 환경을 가리지 못하게 합니다. rc 파일에서 전역으로
  `source /opt/ros/<distro>/setup.*` 하지 **마세요**;
- colcon 워크스페이스를 소싱하고 `cbuild` / `ros2kill`을 정의합니다.

호스트 rc 파일의 영향을 완전히 차단하려면 컨테이너(`.devcontainer` / `.docker`)를
사용하세요.
</details>

## 시스템 ROS 2 (apt / rosdep) — 아직 미검증

<details><summary>실험적 경로</summary>

```bash
# B1  ROS 2 Jazzy 설치 (https://docs.ros.org/en/jazzy/Installation.html) 후:
source /opt/ros/jazzy/setup.bash

# B2  rosdep (각 package.xml의 ROS·apt 의존성)
cd ~/unicorn_ws && rosdep install --from-paths src --ignore-src -r -y

# B3  python 레이어 (conda 경로와 동일한 requirements.txt)
pip install -r src/unicorn-racing-stack/requirements.txt

# B4  빌드
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```
</details>

---

# 컨트롤러 파라미터 튜닝 (Optuna)

튜너는 [`tuning/`](tuning)에 있습니다. 아래 순서로 쓰세요:

1. **`fast_tune.py` — 헤드리스 대량 랩 탐색.** ROS 그래프도 벽시계도 없이
   f1tenth_gym 물리를 CPU 속도로 밟고 컨트롤러를 50 Hz로 직접 호출합니다.
   시간당 수천 trial. **실제 탐색은 여기서 이뤄집니다.**
2. **`validate_best.py` — 실제 스택 검증.** 저장된 best를 전체 스택에 적용해
   동일 지표를 측정합니다. **채택 전 반드시 검증하세요** — 헤드리스 모델은
   전체 스택과 같지 않습니다 ([주의사항](#주의사항--결과를-믿기-전에-읽으세요) 참고).
3. **`tune_controller.py` — 폐루프 튜너.** 실행 중인 시뮬을 상대로 돌립니다.
   충실도가 가장 높지만 훨씬 느립니다. 선택 사항.

> **모든 명령 블록에 `cd`가 포함돼 있습니다.** 어느 디렉토리에 있든 그대로
> 복사해 붙여넣으면 됩니다.

## 파일

| 경로 | 설명 |
|---|---|
| `tuning/tuner_config.yaml` | **수정하는 유일한 파일**: 탐색 공간, 고정값, 목적함수 가중치 |
| `tuning/fast_tune.py` | 헤드리스 탐색 (메인 진입점) |
| `tuning/progress.py` | 스터디의 trial 수 / 속도 / ETA / 현재 best |
| `tuning/validate_best.py` | 저장된 best를 실제 스택에서 재현 |
| `tuning/tune_controller.py` | 폐루프 튜너 (실행 중인 시뮬 필요) |
| `tuning/overnight.sh` | 무인 시뮬 + 폐루프 튜너 실행 |
| `tuning/best_params_<스터디이름>.yaml` | 스터디의 best trial, `--show-best`가 생성 |
| `tuning/journal_fast.log` | Optuna 저장소, 첫 실행 시 생성 (gitignore, 수백 MB까지 증가) |
| `stack_master/config/controller.yaml` | Pure-Pursuit 파라미터 — `--ctrl pp`의 대상 |
| `stack_master/config/controller_map.yaml` | MAP 파라미터 — `--ctrl map`의 대상 |
| `stack_master/maps/<맵>/global_waypoints.json` | trial이 주행하는 레이스라인 + 속도 프로파일 |

## 1. 환경 진입

```bash
cd ~/unicorn_ws
source src/unicorn-racing-stack/unicorn.sh
cd src/unicorn-racing-stack/tuning
```

conda(RoboStack) 전용입니다. **`source /opt/ros/jazzy/setup.bash`를 먼저 하지
마세요** — PATH에 시스템 ROS가 있으면 토픽은 동작하는데 파라미터 서비스가
타임아웃 나는 증상이 생깁니다.

스크립트가 자기 위치로부터 스택 경로를 찾으므로 어느 경로에 클론해도 동작합니다.
필요 시 오버라이드:

```bash
export UNICORN_STACK=/path/to/unicorn-racing-stack   # 트리를 옮긴 경우에만
export UNICORN_MAP=s                                 # 튜닝할 맵 (기본: s)
```

## 2. 스모크 테스트 — 항상 먼저 실행

```bash
cd ~/unicorn_ws/src/unicorn-racing-stack/tuning
python fast_tune.py --ctrl map --smoke
```
```
smoke (map yaml params, scaling 1.0):
  laps ['9.02', '9.02', '9.02', '9.02']  |d| 0.084  weave 0.0064  osc 0.0194
  wall time 2.5s for 3 laps -> 1.2 laps/s/core
```

**현재 yaml 파라미터**를 한 번 평가합니다. 여기서 실패하거나 크래시하면 탐색 전에
그것부터 고치세요. `laps/s/core` × 워커 수가 처리량 예산입니다.

## 3. 탐색 설정 — `tuner_config.yaml`

```yaml
params:                     # enabled: true -> [low, high] 범위에서 탐색
  m_l1:       {enabled: true, low: 0.35, high: 1.00}
  t_clip_min: {enabled: true, low: 0.60, high: 2.50}
frozen:                     # 매 trial 적용, 탐색하지 않음
  "/speed_sector_tuner:Sector0.scaling": 1.0
  curvature_factor: 0.0
  KI: 0.0
objective:
  w_lat_err: 4.0            # 평균 |d| 1 m당 비용
  d_hard_limit: 0.10        # |d| 예산, 초과분에 w_d_over 적용
  w_osc: 64.0               # 주기당 조향 변화량 RMS 1 rad당
  w_weave: 2000.0           # 부호 있는 d 증분 RMS 1 m당 (차체 흔들림)
  w_osc_worst: 40.0         # 위와 같되 최악 ~1초 구간 기준 (코너별)
  w_weave_worst: 200.0
  ou_sigma: 0.025           # 저주파 횡외란 크기 [rad] (0이면 외란 없음)
  ou_laps: 3                # 시드당 측정 랩 수 (시드 3개 -> 비용 2.25배)
  fail_penalty: 120.0       # 충돌 / 정지 / 이탈 / 타임아웃
```

`cost = 랩타임 + w_lat_err·|d| + w_d_over·초과 + w_osc·osc + w_weave·weave +
w_osc_worst·osc_worst + w_weave_worst·weave_worst` — **낮을수록 좋습니다.**

### OU 외란 (v9~)

`ou_sigma > 0`이면 물리 입력단의 **바퀴 조향각에 저주파 외란**(OU 과정, τ=0.7초)을
더합니다. 컨트롤러는 이를 인지하지 못하고 결과만 겪으므로, 외란을 증폭하는
파라미터(짧은 L1, 높은 게인)가 자동으로 벌점을 받습니다.

실차 백에서 보정했습니다 — 실차는 시뮬에 없는 0–2 Hz 횡운동을 보이며
(weave 0.00689 vs 0.00466), 이는 센서 잡음이 아닙니다(3 Hz 이상 성분은 3 mm뿐).
외란이 없으면 튜너가 v7b와 v8을 0.4% 차이로 동급 취급하지만, 넣으면 4.1%로
벌어지고 그게 실차 거동과 일치합니다.

시드(`fast_tune.OU_SEEDS`)는 모든 trial에 동일하게 고정되므로 **목적함수는
결정론적으로 유지**됩니다 — TPE가 운 좋은 난수를 쫓지 않습니다.

현재 MAP 탐색은 17차원입니다. `Sector0.scaling`은 1.0 고정(기하만 튜닝),
조향 다운스케일 밴드 `start/end_scale_speed`는 6.5 / 10.0 고정,
`curvature_factor`는 **양쪽 컨트롤러 모두 0으로 고정**(L1 성형은 chord-cap과
MAP 피드포워드로 비교하므로 교란 요인 제거). PP는 `lat_err_steer_coeff`가
빠져 16차원입니다.

## 4. 탐색 실행

```bash
cd ~/unicorn_ws/src/unicorn-racing-stack/tuning

# ↓↓↓ 이 줄의 이름만 바꾸세요. 아래 명령들이 전부 이 값을 씁니다 ↓↓↓
STUDY=controller_fast_v9_map          # 스터디 이름 (탐색 공간이 바뀌면 새 이름!)

nohup python fast_tune.py --ctrl map --study $STUDY \
      --n-trials 30000 --workers $(( $(nproc) - 2 )) \
      > fast_$STUDY.log 2>&1 &
echo "pid $!"
```

- `STUDY`는 **평범한 bash 변수**입니다. 첫 줄의 값만 바꾸면 이후 명령에 자동
  반영되므로, 매번 이름을 다시 타이핑할 필요가 없습니다.
- `--n-trials`는 **전체 합계**이며 워커에 나뉩니다(`n_trials // workers`씩).
- 코어를 2개 남기세요. 16코어 노트북에서 대략 분당 70–90 trial입니다.
- PP와 MAP을 동시에 돌리려면 스터디 이름과 `--ctrl`을 각각 다르게 주고 워커를
  나누면 됩니다 (예: 각 6개).

**⚠️ 탐색 공간·목적함수·레이스라인·`ggv.csv`·`dynamics.yaml` 중 하나라도 바뀌면
반드시 새 스터디 이름을 쓰세요.** 옛 trial은 다른 물리에서 채점된 값이라 뒤섞이면
순위가 오염됩니다.

## 5. 진행 상황 보기

### 추천: `tcount` 별칭 등록

매번 긴 경로를 치지 않도록 `~/.bashrc`에 넣어두는 걸 **강력히 권장**합니다:

```bash
cat >> ~/.bashrc <<'EOF'

# ── unicorn 튜닝 단축키 ────────────────────────────────────────────────
# tcount            : 저널의 모든 스터디를 한 표에
# tcount9           : v9 스터디 두 개만 (map + pp)
# tcount9 30000     : + 목표 trial까지 남은 시간(ETA)
# tcount <스터디이름> : 특정 스터디 하나만
#
# progress.py는 optuna만 임포트하므로 unicorn.sh 소싱 없이 ~2초에 뜹니다.
# 저널이 수백 MB라 한 번만 열어 모든 스터디를 읽습니다 (스터디마다 호출하면 느림).
tcount() {
    local T=~/unicorn_ws/src/unicorn-racing-stack/tuning
    ~/miniforge3/envs/unicorn/bin/python "$T/progress.py" "$@"
}
tcount9() { tcount controller_fast_v9_map controller_fast_v9_pp "$@"; }
EOF
source ~/.bashrc
```

```bash
tcount9
```
```
study                      trials    ran  fail%   wrk   /min     best    lap     |d|    weave
---------------------------------------------------------------------------------------------
* controller_fast_v9_map      856    616    26%     6   14.3   24.112   9.25  0.0518  0.00468
* controller_fast_v9_pp       357    234    32%     6   14.2   26.795   9.24  0.0531  0.00581

  * = 워커 붙어 있음   (n) = 죽은 워커가 남긴 유령 RUNNING 행   /min = 최근 200개 완료 기준
```

| 컬럼 | 의미 |
|---|---|
| `trials` / `ran` | 전체 / **완주한** trial (실패 제외) |
| `fail%` | 실패율 — 탐색 초기 10~30%는 정상(TPE 탐색 중) |
| `wrk` | 붙어 있는 워커 수. **괄호면 멈춘 스터디의 유령 행** |
| `/min` | **최근 완료 구간** 기준 처리량 (정지 구간 제외) |
| `best`~`weave` | best trial의 cost / 랩타임 / 평균\|d\| / weave |

별칭 없이 쓰려면:

```bash
cd ~/unicorn_ws/src/unicorn-racing-stack/tuning
python progress.py                                          # 모든 스터디
python progress.py controller_fast_v9_map 30000             # 하나 + ETA
```

### 순위를 다르게 보기 (아무것도 덮어쓰지 않음)

```bash
cd ~/unicorn_ws/src/unicorn-racing-stack/tuning
STUDY=controller_fast_v9_map

python fast_tune.py --ctrl map --study $STUDY --show-best --no-apply
python fast_tune.py --ctrl map --study $STUDY --show-best --no-apply --max-d 0.05 --by lap
python fast_tune.py --ctrl map --study $STUDY --plot-best     # 재주행 + 그래프
```

### 중지 / 일시정지 / 재개

저널이 남으므로 **같은 스터디 이름으로 재실행하면 이어서 진행**됩니다.

```bash
pkill -INT  -f 'python fast_tune\.py'   # 종료 (권장, 저널 저장됨)
pkill -STOP -f 'python fast_tune\.py'   # 일시정지 (CPU 반환)
pkill -CONT -f 'python fast_tune\.py'   # 재개 -- STOP 했으면 반드시 실행!
```

`-STOP`은 죽이는 게 아니라 멈추는 것이라, `-CONT`를 잊으면 프로세스가 `T` 상태로
영원히 남습니다. 상태 확인은 자기참조를 피해서:

```bash
ps -eo pid=,stat=,args= | grep '[f]ast_tune\.py'
```

## 6. 결과 채택

```bash
cd ~/unicorn_ws/src/unicorn-racing-stack/tuning
STUDY=controller_fast_v9_map

# 1) 스냅샷 저장 + controller_map.yaml에 실제로 기록
python fast_tune.py --ctrl map --study $STUDY --show-best

# 2) 워커 일시정지 -- 실제 시뮬은 실시간이라 CPU 경합에 취약합니다
pkill -STOP -f 'python fast_tune\.py'

# 3) launch 파일/설정 변경분을 install에 반영
cd ~/unicorn_ws && colcon build --packages-select stack_master

# 4) 터미널 2: 스택 실행 후 'a'를 눌러 arm
#    (정지해 있으면 human 모드입니다 — "시뮬레이션 빠른 실행" 참고)
ros2 launch stack_master race.launch.xml sim:=true map:=s use_map:=true

# 5) 터미널 1: 동일 지표를 전체 스택에서 측정
cd ~/unicorn_ws/src/unicorn-racing-stack/tuning
python validate_best.py --params best_params_$STUDY.yaml --laps 2

# 6) 탐색 재개
pkill -CONT -f 'python fast_tune\.py'
```

실행 중인 스택에서 파라미터를 라이브로 만지고 파일에 굳히려면:

```bash
ros2 param set /controller_manager m_l1 0.42
ros2 param set /controller_manager save_params true     # -> controller_map.yaml
```

## 7. 폐루프 튜너 (선택)

`tune_controller.py`는 이미 실행 중인 스택을 상대로 돌며, 스터디는 스크립트 옆
SQLite(`tuning/optuna_stage1.db`, gitignore)에 저장됩니다:

```bash
# 터미널 2
ros2 launch stack_master race.launch.xml sim:=true map:=s use_map:=true

# 터미널 1
cd ~/unicorn_ws/src/unicorn-racing-stack/tuning
python tune_controller.py --n-trials 60
```

`overnight.sh`(무인 시뮬 + 튜너, flock으로 단일 인스턴스 보장)가 둘을 감싸며,
자기 위치로부터 경로를 유도하므로 어느 경로에 클론해도 수정이 필요 없습니다.

## 주의사항 — 결과를 믿기 전에 읽으세요

- **`--show-best`는 `controller.yaml` / `controller_map.yaml`을 덮어씁니다.**
  출력·스냅샷만 원하면 `--no-apply`를 주세요. 우승 trial의 모든 파라미터와
  `frozen` 값이 함께 기록됩니다.
- **문제가 바뀌면 반드시 새 `--study` 이름을 쓰세요** — 탐색 공간, 목적함수
  가중치, 레이스라인, `ggv.csv`, `dynamics.yaml` 중 무엇이든. 옛 trial은 옛
  물리에서 채점됐으므로 낡은 근거로 순위를 차지합니다. `fast_tune`은 레이스라인을
  시작 시 한 번만 읽습니다.
- **헤드리스 ≠ 전체 스택.** 물리는 100 Hz(브리지는 80 Hz)로 돌고, state_machine /
  로컬 플래너를 거치지 않습니다. 상위 몇 개는 실제 시뮬에서 검증한 뒤 채택하세요.
  (구동 지연 30 ms 모델 `LATENCY_S`는 실차 백으로 검증됐습니다 — 명령→요레이트
  응답 지연이 실차·시뮬 모두 110 ms. `/local_waypoints` 지연도 실측 0.1 ms로
  무시 가능합니다.)
- **CPU 경합은 실제 시뮬 측정만 망칩니다.** `validate_best.py` /
  `tune_controller.py` 전에는 반드시 워커를 `SIGSTOP` 하세요. 반대로 `fast_tune`은
  시뮬레이션 시간으로 돌기 때문에 **헤드리스 튜너끼리는 같이 돌려도 결과가
  안 바뀝니다**(느려질 뿐).
- **`journal_fast.log`는 수백 MB까지 커지고** 모든 워커가 시작 시 이를 재생합니다.
  캠페인 사이에 정리하세요:
  `mv journal_fast.log journal_fast.log.$(date +%F)`
  (이후 옛 스터디는 읽을 수 없게 되지만 `best_params_*.yaml` 스냅샷은 남습니다).
- **헤드리스 단독 실행으로 채점할 수 없는 파라미터**를 `enabled`로 두면 무의미한
  랜덤워크 차원만 늘어납니다: `trailing_*`(상대차와 갭 오차 항 필요),
  `start_speed` / `speed_diff_thres` / `start_curvature_factor`(START 상태;
  `fast_tune`은 항상 `GB_TRACK` 주행), `ftg_*`(FTG 상태).
- **`AEB_thres`(안전)와 `KI`는 절대 탐색하지 마세요.** 현재 적분항에는 클램프도
  조건부 정지도 없어 와인드업이 무제한입니다. `KI`를 열려면 안티와인드업을 먼저
  구현해야 합니다. `steer_gain_for_speed`는 1.0에서 무동작이고 그 위로는 전역
  조향 배수가 되어 MAP 피드포워드 보정과 충돌합니다.
- **`Sector0.scaling`만이 랩타임 상한을 올립니다.** 고정된 속도 프로파일 위에서
  기하를 튜닝하도록 1.0으로 고정돼 있습니다. 올리면 재튜닝이 필요합니다 —
  고속에서 조향 다운스케일 밴드와 `downscale_factor`가 실질 권한을 갖게 됩니다.
- **시뮬 2개 / `overnight.sh` 2개를 동시에 띄우지 마세요.** gym 인스턴스 둘이
  odom을 섞어버려 모든 소비자가 오작동합니다(0.05초 랩, 랩 카운터 중복).

## 시뮬 플랜트 보정 (2026-07-26)

실차 백으로 시뮬을 보정했습니다. **이 값들을 임의로 되돌리면 튜닝 결과가
실차와 어긋납니다.**

| 항목 | 값 | 근거 |
|---|---|---|
| `SIM/dynamics.yaml` `C_Sf` | **3.80** (기존 4.718) | 실측 조향 권한 0.697 (시뮬 0.703). 4.718은 0.8 g에서도 0.874로 과접지 |
| 룩업 테이블 | sim·실차 모두 `UNICORN2-0410` | `SIM_linear`는 타이어 포화가 없어 3.05 g를 가정(실측 1.07 g). 조향을 40% 적게 내 언더스티어 유발 |
| `LATENCY_S` | 0.03 (유지) | 명령→요레이트 지연 실차 110 ms = 시뮬 110 ms |
| `ou_sigma` | 0.025 | 실차 0–2 Hz 횡운동에 보정 |

**OU 외란 모델의 한계 (기록용):** σ는 파라미터셋 **하나**에 대해 스칼라 통계
**둘**(mean\|d\|, weave)만 맞춘 값입니다. 실차 d(s)를 12랩으로 분해하면 체계적 50% /
무작위 50%인데, 그 체계적 성분이 시뮬의 것과 **상관 +0.019**로 사실상 무관합니다
(크기는 비슷하나 틀리는 위치가 다름). 즉 **체계적 모델 오차가 남아 있고 OU가
그것을 덮고 있을 수 있습니다.** 실차 검증 전에 이 모델 위에 제어 구조 변경
(적분항 활성화 등)을 쌓지 마세요.

다만 σ 크기 자체에는 강건합니다 — σ ≥ 0.015 전 구간에서 후보 순위가 동일합니다
(σ = 0에서만 뒤집힙니다).

## Credits

이 스택은
[HMCL-UNIST `unicorn-racing-stack`](https://github.com/hmcl-unist/unicorn-racing-stack)
(ROS 2 Jazzy 포팅 및 레이싱 파이프라인) 위에서 작업한 결과의 스냅샷이며, 그
원류는 [ForzaETH `race_stack`](https://github.com/ForzaETH/race_stack)
(ETH Zürich)입니다. 벤더링된 컴포넌트는 각자의 업스트림에서 왔습니다:

- `race_utils/raycaster` — [jeongsang-ryu/2D-RayCaster](https://github.com/jeongsang-ryu/2D-RayCaster)
- `race_utils/unicorn_gym` — [jeongsang-ryu/unicorn-gym](https://github.com/jeongsang-ryu/unicorn-gym)
- `state_estimation/kiss_icp_localization` — [freshleesh/kiss_icp_localization](https://github.com/freshleesh/kiss_icp_localization)

이 리포는 평탄화된 단일 커밋 스냅샷으로 공개되며, 전체 저작 이력은 위 업스트림
리포에 있습니다.

# id_analyser — ROS2 적용 노트 (2026-07-23 ForzaETH race_stack에서 이식)

출처: https://github.com/ForzaETH/race_stack `system_identification/id_analyser` (MIT).
`SysID_checklist.md` = 원본 `stack_master/checklists/SysID.md`.

## 목적

실차 조향 룩업테이블(`{차량이름}_pacejka_lookup_table.csv`) 생성 파이프라인의
빠져 있던 가운데 조각(rosbag 분석 → Pacejka 피팅 → 테이블 생성).
결과물은 `../steering_lookup/cfg/`에 넣고 `controller.yaml`의 `LU_table`로 선택.

## 워크플로우 (조향 테이블 기준)

1. `add_model.py` — 새 차량 모델 등록 (질량, l_f/l_r, I_z 등 물리 파라미터; `models/<이름>/`)
2. 실차 주행: `id_controller` experiment 5 (정속 + 조향 램프)를
   **여러 속도(2~5 m/s)·좌우 양방향**으로 반복하며 bag 기록.
   필요한 토픽: `/car_state/odom`, `/state_estimation/odom`(yaw rate),
   `/vesc/commands/servo/position`
3. bag들을 `data/<모델>/tire_dynamics/`에 배치
4. `analyse_tires.py` — 정상상태 구간에서 슬립각·횡력 계산, Pacejka B/C/D 최소자승 피팅
   (파일 상단 `model_name_` 수정)
5. `generate_lookup_table.py` — 피팅된 모델로 (조향각 × 속도) → 횡가속 그리드 생성·저장
6. CSV를 `../steering_lookup/cfg/`로 복사, `LU_table` 갱신, `use_map: true`

## ROS1 → ROS2 포팅 상태

- **`helpers/bagloader.py` — 포팅 완료 (2026-07-23)**: bagpy 제거,
  `rosbags.highlevel.AnyReader` 기반으로 재작성. ROS1 `.bag`, ROS2 디렉토리
  (metadata.yaml), `.db3`, `.mcap` 모두 읽음. 인터페이스(`load_bags`,
  dotted 컬럼명, `<topic>.data`, `header.stamp.nsecs` int64 정렬축) 유지 —
  분석 스크립트 무수정 호환. 합성 ROS2 bag으로 시간 정렬까지 검증됨.
  의존성: `pip install rosbags pandas` (unicorn env에 설치됨).
- 토픽 이름 확인: 우리 스택은 `/car_state/odom`(state_estimation 발행) 사용 —
  원본의 `/state_estimation/odom`와 이름이 다르면 field_dict만 맞춰줄 것.
- servo 명령 토픽(`/vesc/commands/servo/position`)은 우리 vesc 드라이버와 동일.
- 나머지(피팅·생성 스크립트)는 순수 numpy/scipy/pandas라 ROS 버전 무관.

## 참고

- 더 새 파이프라인(NN 잔차 보정)은 원본 repo `system_identification/on_track_sys_id`
  (On-Track SysID 논문) — 필요해지면 같은 방식으로 이식.
- 기존 우리 테이블들(UNICORN*)과 포맷 동일: 첫 행 = 속도 그리드, 첫 열 = 조향각,
  셀 = 정상상태 횡가속.

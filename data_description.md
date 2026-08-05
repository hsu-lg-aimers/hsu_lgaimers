# 데이터 설명서

## [배포용 데이터 구조]

```text
open.zip

baseline_submit.zip : 베이스라인 코드 기반 리더보드 제출 파일(zip) 예시 (참고용)
data/
  - train.csv : 학습 입력 및 정답 데이터 (1,475,092행 x 49컬럼)

  - test.csv : 평가 입력 데이터 (형식 확인용 5건 샘플, 48컬럼)

  - sample_submission.csv : 제출 양식 파일 (형식 확인용 5행 x 2컬럼)

  - trackman_history.csv : 2019~2024년 Trackman 과거 로그 (1,793,078행 x 30컬럼)

data_description.md : 데이터 설명서
```

※ `test.csv`는 배포본에 형식 확인용 5건만 포함됩니다. 실제 평가 데이터는 비공개이며, 제출용 파일(zip)을 리더보드 평가에 제출하면 평가 서버에서 동일한 경로와 동일한 컬럼 구조의 실제 평가 데이터로 교체되어 처리됩니다.

※ `sample_submission.csv` 역시 형식 확인용 5건 샘플입니다. 실제 평가 시에는 평가 서버의 `test.csv` 행 수와 동일한 제출 양식을 생성해 제출해야 합니다.

※ 모든 데이터 파일은 CSV 형식입니다. `trackman_history.csv`는 메인 학습/평가 데이터와 1:1로 결합하는 정답 테이블이 아니라, 참가자가 과거 이력 기반 피처를 만들 때 참고할 수 있는 로그 데이터입니다.

## [세부 설명]

### 1) 학습/추론 데이터: `train.csv` · `test.csv`

각 행은 한 개의 투구 시점 상태를 나타냅니다. 참가자는 투구 직전까지 알 수 있는 경기 상황, 선수/팀 정보, 과거 이력 피처를 바탕으로 해당 투구의 제구 성공 확률을 예측합니다.

`train.csv`와 `test.csv`의 입력 피처 구조는 동일합니다. 단, `train.csv`에는 학습 정답인 `control_success`가 포함되고, `test.csv`에는 정답 컬럼이 포함되지 않습니다.

#### 기본 식별자 및 경기 정보

| 컬럼 | 설명 |
| --- | --- |
| `row_id` | 샘플 고유 식별자입니다. 제출 파일과 매칭하는 데 사용합니다. |
| `season` | 시즌 연도입니다. |
| `game_month` | 경기 월입니다. |
| `game_dayofweek` | 경기 요일입니다. 월요일은 0, 일요일은 6입니다. |
| `inning` | 투구 직전 이닝입니다. |
| `top_bottom` | 초/말 구분입니다. `T`는 초, `B`는 말을 의미합니다. |
| `game_type` | 경기 유형 코드입니다. |

#### 투구 직전 카운트 및 점수 상황

| 컬럼 | 설명 |
| --- | --- |
| `balls_before` | 투구 직전 볼 카운트입니다. |
| `strikes_before` | 투구 직전 스트라이크 카운트입니다. |
| `outs_before` | 투구 직전 아웃 카운트입니다. |
| `run_top_before` | 투구 직전 초 공격 팀의 점수입니다. |
| `run_bot_before` | 투구 직전 말 공격 팀의 점수입니다. |
| `run_total_before` | 투구 직전 양 팀 합산 점수입니다. |
| `score_diff_home` | 투구 직전 홈 팀 기준 점수 차입니다. |
| `score_diff_pitcher_team` | 투구 직전 투수 소속 팀 기준 점수 차입니다. |

#### 주자 및 상황 중요도

| 컬럼 | 설명 |
| --- | --- |
| `runner_on_1b` | 투구 직전 1루 주자 여부입니다. `1`은 있음, `0`은 없음을 의미합니다. |
| `runner_on_2b` | 투구 직전 2루 주자 여부입니다. `1`은 있음, `0`은 없음을 의미합니다. |
| `runner_on_3b` | 투구 직전 3루 주자 여부입니다. `1`은 있음, `0`은 없음을 의미합니다. |
| `num_runners_on` | 투구 직전 출루 주자 수입니다. |
| `base_state` | 투구 직전 주자 상황입니다. `___`=주자 없음, `1__`=1루, `_2_`=2루, `__3`=3루, `12_`=1/2루, `1_3`=1/3루, `_23`=2/3루, `123`=만루입니다. |
| `home_win_expectancy` | 투구 직전 경기 상황에서 홈 팀의 기대 승률입니다. 0~100 범위의 값입니다. |
| `away_win_expectancy` | 투구 직전 경기 상황에서 원정 팀의 기대 승률입니다. 0~100 범위의 값입니다. |
| `li` | 투구 직전 상황 중요도 지표입니다. 값이 클수록 경기 흐름에 미치는 영향이 큰 상황을 의미합니다. |

#### 선수 및 팀 정보

| 컬럼 | 설명 |
| --- | --- |
| `pitcher_id` | 투수 익명 ID입니다. |
| `batter_id` | 타자 익명 ID입니다. |
| `pitcher_hand` | 투수의 좌우 유형 코드입니다. |
| `batter_hand` | 타자의 좌우 유형 코드입니다. |
| `pitcher_team_id` | 투수 소속 팀 ID입니다. |
| `batter_team_id` | 타자 소속 팀 ID입니다. |

#### 투구 직전 기준 과거 이력 피처

`asof_*` 컬럼은 해당 행의 투구 직전까지 확인 가능한 과거 기록으로 사전 계산된 피처입니다. 현재 투구 이후에 확정되는 정보는 사용하지 않았습니다.

| 컬럼 | 설명 |
| --- | --- |
| `asof_pitcher_n` | 해당 투구 직전까지 해당 투수의 누적 투구 수입니다. |
| `asof_pitcher_success_rate` | 해당 투구 직전까지 해당 투수의 제구 성공률입니다. |
| `asof_pitcher_reverse_rate` | 해당 투구 직전까지 해당 투수의 의도 반대성 투구 비율입니다. |
| `asof_pitcher_middle_rate` | 해당 투구 직전까지 해당 투수의 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_ball_rate` | 해당 투구 직전까지 해당 투수의 볼성 결과 비율입니다. |
| `asof_pitcher_strike_rate` | 해당 투구 직전까지 해당 투수의 스트라이크성 결과 비율입니다. |
| `asof_pitcher_prev1_game_success_rate` | 해당 투수의 직전 1경기 제구 성공률입니다. |
| `asof_pitcher_prev3_game_success_rate` | 해당 투수의 직전 3경기 제구 성공률입니다. |
| `asof_pitcher_prev5_game_success_rate` | 해당 투수의 직전 5경기 제구 성공률입니다. |
| `asof_pitcher_prev1_game_middle_rate` | 해당 투수의 직전 1경기 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_prev3_game_middle_rate` | 해당 투수의 직전 3경기 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_prev5_game_middle_rate` | 해당 투수의 직전 5경기 가운데 또는 위험 코스 비율입니다. |
| `asof_batter_n` | 해당 투구 직전까지 해당 타자가 상대한 누적 투구 수입니다. |
| `asof_batter_success_rate` | 해당 투구 직전까지 해당 타자가 상대한 투구의 제구 성공률입니다. |
| `asof_batter_middle_rate` | 해당 투구 직전까지 해당 타자가 상대한 투구의 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_pitchmix_n` | 해당 투구 직전까지 해당 투수의 구종 사용 이력 표본 수입니다. |
| `asof_pitcher_fastball_rate` | 해당 투구 직전까지 해당 투수의 fastball 계열 사용 비율입니다. |
| `asof_pitcher_breaking_rate` | 해당 투구 직전까지 해당 투수의 breaking 계열 사용 비율입니다. |
| `asof_pitcher_offspeed_rate` | 해당 투구 직전까지 해당 투수의 offspeed 계열 사용 비율입니다. |

※ 표본 수가 0인 경우 일부 rate 컬럼은 결측값일 수 있습니다. 이런 cold-start 상황의 결측 처리, smoothing, fallback 전략은 참가자가 자유롭게 설계할 수 있습니다.

### 2) 학습 정답 데이터: `train.csv`의 `control_success`

`train.csv`에는 아래 정답 컬럼이 포함됩니다.

| 컬럼 | 설명 |
| --- | --- |
| `control_success` | 예측 대상입니다. `1`은 제구 성공, `0`은 제구 실패를 의미합니다. |

`control_success`는 운영 기준에 따라 산출된 제구 성공 여부입니다. Target 산출에 사용되는 현재 투구의 사후 정보는 입력 피처로 제공되지 않습니다.

### 3) 과거 Trackman 로그: `trackman_history.csv`

`trackman_history.csv`는 2019~2024년 Trackman 과거 로그입니다. 2025년 Trackman 데이터는 제공되지 않습니다.

참가자는 이 파일을 이용해 과거 투구 특성, 구종 특성, 투수 단위 요약값 등 추가 피처를 만들 수 있습니다. 단, 이 파일은 `train.csv` 또는 `test.csv`와 1:1로 직접 결합되는 테이블이 아니며, 평가 시점 이후 정보를 포함하는 방식으로 사용할 수 없습니다.

| 컬럼 | 설명 |
| --- | --- |
| `trackman_id` | Trackman 과거 로그의 행 식별자입니다. |
| `season` | Trackman 투구가 속한 시즌입니다. 2019~2024만 포함됩니다. |
| `game_date` | Trackman 투구의 경기 날짜입니다. |
| `game_month` | Trackman 투구의 경기 월입니다. |
| `game_dayofweek` | Trackman 투구의 경기 요일입니다. 월요일은 0, 일요일은 6입니다. |
| `trackman_game_id` | Trackman 기준 경기 ID입니다. 메인 데이터의 `row_id`와 직접 대응하지 않습니다. |
| `pitch_no` | Trackman 기준 경기 내 투구 번호입니다. |
| `inning` | Trackman 로그의 이닝입니다. |
| `top_bottom` | Trackman 로그의 초/말 표기입니다. |
| `balls_before` | 투구 직전 볼 카운트입니다. |
| `strikes_before` | 투구 직전 스트라이크 카운트입니다. |
| `outs_before` | 투구 직전 아웃 카운트입니다. |
| `pitch_of_pa` | 해당 타석에서의 투구 순번입니다. |
| `pitcher_trackman_id` | Trackman 기준 투수 ID입니다. |
| `batter_trackman_id` | Trackman 기준 타자 ID입니다. |
| `pitcher_hand` | 투수의 좌우 유형 코드입니다. |
| `batter_hand` | 타자의 좌우 유형 코드입니다. |
| `pitcher_team` | 투수 소속 팀입니다. |
| `batter_team` | 타자 소속 팀입니다. |
| `tagged_pitch_type` | 수동 또는 태깅 기반 구종명입니다. |
| `auto_pitch_type` | 자동 분류 기반 구종명입니다. |
| `pitch_type_group` | 구종을 `fastball`, `breaking`, `offspeed`, `other`로 단순화한 구종군입니다. |
| `rel_speed` | 릴리스 시점의 구속입니다. |
| `spin_rate` | 투구 회전수입니다. |
| `induced_vert_break` | 유도 수직 무브먼트입니다. |
| `horz_break` | 수평 무브먼트입니다. |
| `extension` | 릴리스 확장 거리입니다. |
| `rel_height` | 릴리스 높이입니다. |
| `rel_side` | 릴리스 좌우 위치입니다. |
| `zone_speed` | 홈플레이트 근처 구속입니다. |

### 4) 모델 추론 결과 양식 파일: `sample_submission.csv`

| 컬럼 | 설명 |
| --- | --- |
| `row_id` | 평가 데이터 `test.csv`의 샘플 식별자입니다. |
| `control_success` | 예측값입니다. 해당 투구가 제구에 성공할 확률을 0 이상 1 이하의 실수로 입력합니다. |

※ 제출 시 `row_id` 값은 평가 서버에서 제공되는 `test.csv`와 정확히 일치해야 합니다.

### 5) 평가 데이터 예측 원칙

평가 데이터의 각 행은 독립적으로 예측해야 합니다. 평가 서버에서 실제 `test.csv` 전체가 주어지더라도, 참가자는 `test.csv`의 다른 행을 이용해 현재 행의 피처를 만들 수 없습니다.

금지되는 예시는 다음과 같습니다.

- `test.csv` 내부 행들을 이용한 선수별, 팀별, 월별 누적 통계
- `test.csv` 내부 빈도값 또는 분포 통계
- `test.csv` 내부 target encoding
- `test.csv` 행 순서 기반 rolling 또는 expanding feature
- 평가 데이터 전체를 보고 만든 사후 보정값

운영 측에서 제공한 `asof_*` 컬럼은 각 행의 투구 직전 시점까지의 과거 기록만으로 계산된 공식 입력 피처이므로 사용할 수 있습니다.

### 6) 사용 금지 정보

공정한 평가를 위해 다음 정보는 입력으로 사용할 수 없습니다.

- 현재 투구 이후에 확정되는 모든 정보
- 현재 투구의 실제 위치 또는 코스 정보
- 현재 투구의 실제 판정, 결과, 제구 성공 여부
- 현재 투구의 실제 구종
- 현재 투구의 Trackman 측정값
- 2025년 Trackman 데이터
- 평가 데이터 내부의 다른 행을 이용해 만든 누적, 빈도, 분포, rolling, target encoding 피처

제공된 `train.csv`, 평가 환경의 `test.csv`, 2019~2024년 `trackman_history.csv`, 그리고 대회 규칙상 허용되는 외부 데이터만 사용할 수 있습니다.

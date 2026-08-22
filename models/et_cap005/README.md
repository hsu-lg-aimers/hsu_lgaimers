# et_cap005

현재까지 실제 제출 점수 기준으로 가장 높았던 모델입니다.

- 실제 제출 점수: 939
- 로컬 2024 holdout BSS: 780.291
- 피처 수: 138
- 모델 파일: `model/model.pkl`
- 추론 스크립트: `script.py`

## 핵심 구성

이 모델은 `LightGBMRegressor + CatBoostClassifier + ExtraTreesClassifier` 3-way 앙상블입니다.

각 모델의 예측값을 validation Brier score가 최소가 되도록 블렌딩합니다.

$$
\hat{p}
= w_{lgbm}\hat{p}_{lgbm}
+ w_{cat}\hat{p}_{cat}
+ w_{et}\hat{p}_{et}
$$

저장된 최종 블렌딩 가중치는 다음과 같습니다.

| Model | Weight |
|---|---:|
| LightGBMRegressor | 0.1594167599 |
| CatBoostClassifier | 0.7905832401 |
| ExtraTreesClassifier | 0.0500000000 |

ExtraTrees는 validation에서는 도움이 되었지만 제출 점수에서는 과적합 위험이 있어서 최대 비중을 `0.05`로 제한했습니다.

## 모델별 설정

### LightGBMRegressor

- objective: `regression`
- metric: `l2`
- n_estimators: 900
- learning_rate: 0.035
- num_leaves: 31
- min_child_samples: 140
- subsample: 0.85
- colsample_bytree: 0.85

분류 모델이 아니라 regression으로 확률값을 직접 예측한 뒤 `[0, 1]` 범위로 clip합니다.

### CatBoostClassifier

- loss_function: `Logloss`
- iterations: 900
- learning_rate: 0.035
- depth: 6
- l2_leaf_reg: 5.0

범주형 피처는 CatBoost에 `cat_features`로 직접 전달합니다.

### ExtraTreesClassifier

- n_estimators: 350
- criterion: `log_loss`
- max_features: 0.70
- min_samples_leaf: 60
- min_samples_split: 120
- bootstrap: False

ExtraTrees는 CPU 전용입니다. GPU 옵션을 줘도 ExtraTrees는 CPU에서 학습됩니다.

## 피처 구성

기본 train 피처에 다음 파생 피처를 추가합니다.

- count 상태: `count_state`, `is_full_count`, `is_pitcher_ahead`, `is_hitter_ahead`, `is_even_count`
- 주자/득점권 상태: `has_runner`, `has_risp`, `bases_loaded`
- 경기 상황: `pressure`, `late_inning`, `early_inning`, 점수차 절댓값
- 기대 승률 방향 변환: `pitcher_team_win_expectancy`, `batter_team_win_expectancy`
- 누적 경험량 로그 변환: `log_pitcher_n`, `log_batter_n`, `log_pitchmix_n`
- asof rate smoothing: smoothing strength `50`, `200`, `800`
- 최근 경기 delta: 최근 1/3/5경기 투수 성공률 및 middle rate 차이
- 투수/타자 상대 차이: batter rate - pitcher rate
- 구종 비율 파생: fastball-breaking 차이, non-fastball 중 offspeed 비중

Trackman 피처는 아래 key로 집계한 stable prior를 사용합니다.

```text
pitcher_hand, batter_hand, balls_before, strikes_before, outs_before
```

Trackman에서 사용하는 원본 물리량은 다음입니다.

```text
rel_speed, spin_rate, induced_vert_break, horz_break,
extension, rel_height, rel_side, zone_speed
```

Trackman prior에는 위 물리량 평균, 상황별 pitch type rate, smoothing reliability가 포함됩니다.

## Calibration

앙상블 raw prediction에 logit calibration을 적용합니다.

$$
p_{cal}
= \sigma(a \cdot \text{logit}(p_{raw}) + b)
$$

저장된 calibration 값은 다음과 같습니다.

| Parameter | Value |
|---|---:|
| scale \(a\) | 1.0250378579 |
| bias \(b\) | -0.0424532566 |

이 calibration은 2024 holdout에서 Brier score를 최소화하도록 학습했습니다.

## Validation 결과

2024 시즌을 holdout으로 두고, 2023년까지의 train 데이터와 2023년까지의 Trackman prior로 검증했습니다.

| Prediction | AUC | Brier | BSS | AP |
|---|---:|---:|---:|---:|
| LGBM raw | 0.546528 | 0.248248350 | 623.913 | 0.527820 |
| CatBoost raw | 0.548696 | 0.248003488 | 721.933 | 0.530167 |
| ExtraTrees raw | 0.547075 | 0.248101496 | 682.700 | 0.528099 |
| Ensemble raw | 0.549154 | 0.247971329 | 734.807 | 0.530562 |
| Ensemble calibrated | 0.549154 | 0.247857705 | 780.291 | 0.530562 |

로컬 BSS는 아래 식으로 계산합니다.

$$
BSS
= 100000 \times
\left(
1 - \frac{\text{Brier}}{\bar{y}(1-\bar{y})}
\right)
$$

## 실행 방법

학습:

```bash
python train_model.py
```

CatBoost를 GPU로 학습:

```bash
python train_model.py --lgbm-device cpu --catboost-device gpu --gpu-device 0
```

검증만 실행하고 모델 저장 생략:

```bash
python train_model.py --validate-only
```

추론:

```bash
python script.py
```

## 제출/공유 시 주의

Git에는 보통 아래 3개 파일만 올리는 것을 권장합니다.

```text
train_model.py
script.py
requirements.txt
```

`model/model.pkl`은 크기가 커서 Git push나 제출 환경에서 문제가 될 수 있습니다. 제출 플랫폼에는 필요한 방식에 맞춰 모델 파일을 별도로 포함하거나, 플랫폼 규칙에 맞춰 업로드해야 합니다.

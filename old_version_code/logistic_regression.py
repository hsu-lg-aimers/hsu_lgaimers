from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
# 데이터 불러오기
df = train.copy()

drop_cols = [
    "control_success",
    "row_id",
    "pitcher_id",
    "batter_id"
]

# target 분리
X = df.drop(columns=drop_cols)
y = df["control_success"]


# 1. 범주형 컬럼 먼저 정의
categorical_cols = [
    "top_bottom",
    "game_type",
    "base_state",
    "game_dayofweek",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id"
]

# 2. 나머지는 numeric
numeric_cols = [
    col for col in X.columns
    if col not in categorical_cols
]


# 숫자형 전처리
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 범주형 전처리
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 전처리 통합
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])


# Logistic Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss

# pitcher_id를 그룹으로 사용
groups = df["pitcher_id"]

# 80% train / 20% validation
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=40
)

train_idx, valid_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_valid = X.iloc[valid_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]

# 확인
train_pitchers = set(groups.iloc[train_idx])
valid_pitchers = set(groups.iloc[valid_idx])

print("Train shape:", X_train.shape)
print("Valid shape:", X_valid.shape)
print("Train pitchers:", len(train_pitchers))
print("Valid pitchers:", len(valid_pitchers))
print("겹치는 투수 수:", len(train_pitchers & valid_pitchers))

# 학습
model.fit(X_train, y_train)

# 1일 확률 예측
pred_proba = model.predict_proba(X_valid)[:, 1]

# Brier Score
brier = brier_score_loss(y_valid, pred_proba)

print("Brier Score:", brier)

# train 데이터 평균 제구 성공률
r = 0.5237659752747625

# 평균 제구율 Brier Score
baseline_brier = r * (1 - r)

# 대회 Score 계산
score = max(
    0,
    100000 * (1 - brier / baseline_brier)
)

print(f"Brier Score: {brier:.6f}")
print(f"Baseline Brier Score: {baseline_brier:.6f}")
print(f"예상 대회 Score: {score:.2f}")
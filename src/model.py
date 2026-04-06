# ================================
# HOUSE PRICE PREDICTION (KAGGLE LEVEL)
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import xgboost as xgb


# ================================
# 1. LOAD DATA
# ================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# ================================
# 2. SAVE ID
# ================================
test_ids = test["Id"]


# ================================
# 3. TARGET TRANSFORM
# ================================
y = np.log1p(train["SalePrice"])

# Drop target
train.drop(["SalePrice"], axis=1, inplace=True)


# ================================
# 4. COMBINE DATA
# ================================
all_data = pd.concat([train, test], axis=0).reset_index(drop=True)


# ================================
# 5. HANDLE MISSING VALUES
# ================================
# Numerical -> median
num_cols = all_data.select_dtypes(include=["int64", "float64"]).columns
all_data[num_cols] = all_data[num_cols].fillna(all_data[num_cols].median())

# Categorical -> "None"
cat_cols = all_data.select_dtypes(include=["object"]).columns
all_data[cat_cols] = all_data[cat_cols].fillna("None")


# ================================
# 6. FEATURE ENGINEERING 🔥
# ================================
# Total square feet
all_data["TotalSF"] = (
    all_data["TotalBsmtSF"] +
    all_data["1stFlrSF"] +
    all_data["2ndFlrSF"]
)

# Total bathrooms
all_data["TotalBath"] = (
    all_data["FullBath"] +
    0.5 * all_data["HalfBath"] +
    all_data["BsmtFullBath"] +
    0.5 * all_data["BsmtHalfBath"]
)

# House age
all_data["HouseAge"] = 2026 - all_data["YearBuilt"]

# Remodel age
all_data["RemodAge"] = 2026 - all_data["YearRemodAdd"]


# ================================
# 7. ENCODE CATEGORICAL
# ================================
for col in cat_cols:
    lbl = LabelEncoder()
    all_data[col] = lbl.fit_transform(all_data[col])


# ================================
# 8. SPLIT BACK
# ================================
train = all_data[:len(y)]
test = all_data[len(y):]


# ================================
# 9. MODEL (XGBOOST)
# ================================
model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


# ================================
# 10. CROSS VALIDATION
# ================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = np.sqrt(-cross_val_score(
    model,
    train,
    y,
    scoring="neg_mean_squared_error",
    cv=kf
))

print("CV RMSE:", scores.mean())


# ================================
# 11. TRAIN FINAL MODEL
# ================================
model.fit(train, y)


# ================================
# 12. PREDICTIONS
# ================================
preds = model.predict(test)

# Reverse log
preds = np.expm1(preds)


# ================================
# 13. SUBMISSION FILE
# ================================
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": preds
})

submission.to_csv("submission.csv", index=False)

print("✅ Submission file created: submission.csv")

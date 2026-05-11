"""
══════════════════════════════════════════════════════════════
   SUPPLY CHAIN DELAY PREDICTION — UPGRADED MODEL
   Model: XGBoost Regressor
   Goal: better predictive power + stable inference features
══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import joblib
import time

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n🚀 TRAINING UPGRADED MODEL\n")
start = time.time()

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("cleaned_dataset_balanced.csv")
supplier_stats = pd.read_csv("supplier_profiles_enriched.csv")

# =========================
# DATE FEATURES (KEEP SIMPLE BUT USEFUL)
# =========================
df["order_date"] = pd.to_datetime(df["order_date"])

df["month"] = df["order_date"].dt.month
df["day_of_week"] = df["order_date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df.drop(columns=["order_date", "month", "day_of_week"], inplace=True)

# =========================
# SUPPLIER HISTORY FEATURES (IMPORTANT UPGRADE)
# =========================
df = df.sort_values(["supplier_id"])

# rolling delay behavior per supplier
df["supplier_avg_delay_7"] = df.groupby("supplier_id")["delay_gap"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)

df["supplier_std_delay_7"] = df.groupby("supplier_id")["delay_gap"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=2).std()
).fillna(0)

df["supplier_trend"] = df.groupby("supplier_id")["delay_gap"].transform(
    lambda x: x.shift(1).diff().rolling(5, min_periods=1).mean()
).fillna(0)

# delay streak (very powerful feature)
df["late_streak"] = df.groupby("supplier_id")["late_delivery_risk"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum()
)

# =========================
# MERGE SUPPLIER STATS
# =========================
df = df.merge(supplier_stats, on="supplier_id", how="left")

# =========================
# CORE FEATURES (KEEP BUT CLEAN)
# =========================
df["supplier_vs_scheduled"] = df["avg_days_real"] - df["days_scheduled"]
df["scheduled_x_avg"] = df["days_scheduled"] * df["avg_days_real"]
df["supplier_consistency"] = 1 / (df["std_delay"] + 1)

df["scheduled_vs_avg_ratio"] = df["days_scheduled"] / (df["avg_days_real"] + 1)

# =========================
# TARGET
# =========================
TARGET = "delay_gap"

df = df.drop(columns=["days_real", "delivery_status", "late_delivery_risk"], errors="ignore")

# =========================
# SPLIT
# =========================
X = df.drop(columns=[TARGET])
y = df[TARGET]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# IMPUTER
# =========================
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# =========================
# MODEL (IMPROVED TUNING)
# =========================
model = XGBRegressor(
    n_estimators=700,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.05,
    reg_lambda=1.2,
    random_state=42,
    n_jobs=-1
)

print("🔥 TRAINING...\n")

model.fit(X_train, y_train)

preds = np.clip(model.predict(X_test), -2, 10)

# =========================
# METRICS
# =========================
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

def acc(y, p, n):
    return np.mean(np.abs(y - p) <= n) * 100

print("\n════════ RESULTS ════════")
print(f"MAE        : {mae:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R² Score   : {r2:.4f}")
print(f"±1 day acc : {acc(y_test, preds, 1):.2f}%")
print(f"±2 day acc : {acc(y_test, preds, 2):.2f}%")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "best_regressor.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("\n💾 SAVED MODEL + FEATURES")
print(f"⏱ TIME: {round(time.time() - start, 2)} sec")
print("✅ DONE")
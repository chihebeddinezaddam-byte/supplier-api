import pandas as pd
import numpy as np
import joblib

# ══════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════
model = joblib.load("artifacts/best_regressor.pkl")
imputer = joblib.load("artifacts/imputer.pkl")
feature_names = joblib.load("artifacts/feature_names.pkl")


# ══════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════
def validate_history(history):
    return len(history) >= 5


# ══════════════════════════════════════════════════════
# CORE FEATURE ENGINEERING (IMPROVED)
# ══════════════════════════════════════════════════════
def build_features(history):

    df = pd.DataFrame(history)

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["expected_delivery_date"] = pd.to_datetime(df["expected_delivery_date"])
    df["actual_delivery_date"] = pd.to_datetime(df["actual_delivery_date"])

    df = df.sort_values("order_date").reset_index(drop=True)

    # ─────────────────────────────
    # TARGET
    # ─────────────────────────────
    df["delay_gap"] = (
        df["actual_delivery_date"] -
        df["expected_delivery_date"]
    ).dt.days

    # ─────────────────────────────
    # TIME FEATURES
    # ─────────────────────────────
    df["month"] = df["order_date"].dt.month
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df.drop(columns=["order_date", "month", "day_of_week"], inplace=True)

    # ─────────────────────────────
    # LAG FEATURES
    # ─────────────────────────────
    prev = df["delay_gap"].shift(1)

    df["lag_1"] = prev
    df["lag_2"] = prev.shift(1)
    df["lag_3"] = prev.shift(2)

    df["rolling_mean_3"] = prev.rolling(3, min_periods=1).mean()
    df["rolling_mean_7"] = prev.rolling(7, min_periods=1).mean()
    df["rolling_std_7"] = prev.rolling(7, min_periods=2).std().fillna(0)

    df["delay_trend"] = prev.diff().rolling(5, min_periods=1).mean().fillna(0)

    df["late_streak"] = (prev > 0).rolling(5, min_periods=1).sum().fillna(0)

    # ─────────────────────────────
    # BEHAVIOR FEATURES (🔥 NEW IMPORTANT PART)
    # ─────────────────────────────
    df["late_rate"] = (prev > 0).expanding().mean().fillna(0)

    df["std_delay"] = prev.expanding().std().fillna(0)

    df["avg_delay_gap"] = prev.expanding().mean().fillna(0)

    df["positive_ratio"] = (prev > 0).expanding().mean().fillna(0)

    df["negative_ratio"] = (prev < 0).expanding().mean().fillna(0)

    df["consistency_index"] = 1 / (df["std_delay"] + 1)

    df["severity_score"] = df["late_rate"] * df["std_delay"]

    df["recent_vs_global"] = df["rolling_mean_3"] - df["avg_delay_gap"]

    # ─────────────────────────────
    # FINAL FEATURE ROW
    # ─────────────────────────────
    X = df.iloc[[-1]].drop(columns=["delay_gap"])
    X = X.reindex(columns=feature_names, fill_value=0)

    return X, df


# ══════════════════════════════════════════════════════
# STABILITY SCORE (NEW CORE LOGIC)
# ══════════════════════════════════════════════════════
def compute_stability(df):

    late_rate = df["late_rate"].iloc[-1]
    std_delay = df["std_delay"].iloc[-1]

    score = (1 - late_rate) * 0.65 + (1 / (std_delay + 1)) * 0.35

    return score


# ══════════════════════════════════════════════════════
# ROUTER (FIXED LOGIC)
# ══════════════════════════════════════════════════════
def select_model(stability_score):

    if stability_score > 0.75:
        return "RULE"

    elif stability_score > 0.45:
        return "STAT"

    else:
        return "ML"


# ══════════════════════════════════════════════════════
# RULE ENGINE
# ══════════════════════════════════════════════════════
def rule_prediction(df):
    return df["delay_gap"].mean()


# ══════════════════════════════════════════════════════
# STAT ENGINE (IMPROVED)
# ══════════════════════════════════════════════════════
def stat_prediction(df):
    return df["delay_gap"].tail(3).mean()


# ══════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════
def predict_delay(history):

    if not validate_history(history):
        return {
            "error": "Not enough history",
            "message": "Minimum 5 orders required"
        }

    X, df = build_features(history)

    stability = compute_stability(df)
    mode = select_model(stability)

    # ─────────────────────────────
    # EXECUTION
    # ─────────────────────────────
    if mode == "RULE":
        pred = rule_prediction(df)

    elif mode == "STAT":
        pred = stat_prediction(df)

    else:
        X_imp = imputer.transform(X)
        pred = float(model.predict(X_imp)[0])

    # ─────────────────────────────
    # CLAMP
    # ─────────────────────────────
    pred = max(-2, min(10, pred))

    # ─────────────────────────────
    # INTERPRETATION
    # ─────────────────────────────
    if pred <= 0.5:
        interpretation = "On time / early"
    elif pred <= 2:
        interpretation = f"{pred:.1f} days slight delay"
    else:
        interpretation = f"{pred:.1f} days late"

    # ─────────────────────────────
    # RISK (BASED ON REAL STABILITY)
    # ─────────────────────────────
    if stability > 0.75:
        risk = "LOW"
    elif stability > 0.45:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "predicted_delay_gap": round(pred, 2),
        "interpretation": interpretation,
        "risk_level": risk,
        "model_used": mode,
        "stability_score": round(float(stability), 3)
    }
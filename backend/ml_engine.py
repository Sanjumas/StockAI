"""
ml_engine.py — All machine learning logic lives here.
Uses scikit-learn only. No external API calls.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import json
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")
os.makedirs(MODEL_DIR, exist_ok=True)


# ──────────────────────────────────────────────
#  Feature Engineering
# ──────────────────────────────────────────────

def build_features(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns [sku, quantity, sold_at],
    compute per-SKU feature vectors for ML training/prediction.
    """
    if sales_df.empty:
        return pd.DataFrame()

    sales_df = sales_df.copy()
    sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], format="mixed", dayfirst=False)
    sales_df["date"] = sales_df["sold_at"].dt.date

    daily = (
        sales_df.groupby(["sku", "date"])["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "daily_qty"})
    )

    features = []
    for sku, grp in daily.groupby("sku"):
        grp = grp.sort_values("date")
        qty = grp["daily_qty"].values
        n = len(qty)

        # Rolling windows
        recent_7 = qty[-7:].mean() if n >= 7 else qty.mean()
        recent_14 = qty[-14:].mean() if n >= 14 else qty.mean()
        recent_30 = qty[-30:].mean() if n >= 30 else qty.mean()
        overall_mean = qty.mean()
        overall_max = qty.max()
        overall_std = qty.std() if n > 1 else 0.0

        # Trend: slope of last 14 days vs previous 14 days
        if n >= 28:
            trend = qty[-14:].mean() - qty[-28:-14].mean()
        elif n >= 14:
            trend = qty[-7:].mean() - qty[:7].mean()
        else:
            trend = 0.0

        # Velocity acceleration
        accel = recent_7 - recent_14 if n >= 14 else 0.0

        features.append({
            "sku": sku,
            "avg_daily_7d": recent_7,
            "avg_daily_14d": recent_14,
            "avg_daily_30d": recent_30,
            "avg_daily_overall": overall_mean,
            "max_daily": overall_max,
            "std_daily": overall_std,
            "trend": trend,
            "acceleration": accel,
            "data_days": n,
        })

    return pd.DataFrame(features)


def compute_target(sales_df: pd.DataFrame, horizon_days: int = 7) -> pd.Series:
    """
    Target variable: total demand in the next `horizon_days` days.
    We simulate this by using the last horizon_days of data as the target.
    """
    sales_df = sales_df.copy()
    sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], format="mixed", dayfirst=False)

    targets = {}
    for sku, grp in sales_df.groupby("sku"):
        grp = grp.sort_values("sold_at")
        cutoff = grp["sold_at"].max() - pd.Timedelta(days=horizon_days)
        target_qty = grp[grp["sold_at"] > cutoff]["quantity"].sum()
        targets[sku] = max(target_qty, 0)

    return pd.Series(targets, name="demand_next_7d")


# ──────────────────────────────────────────────
#  Model Registry
# ──────────────────────────────────────────────

MODELS = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "ridge_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "random_forest": Pipeline([
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ]),
    "gradient_boosting": Pipeline([
        ("model", GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)),
    ]),
}

FEATURE_COLS = [
    "avg_daily_7d", "avg_daily_14d", "avg_daily_30d",
    "avg_daily_overall", "max_daily", "std_daily",
    "trend", "acceleration", "data_days"
]


# ──────────────────────────────────────────────
#  Train
# ──────────────────────────────────────────────

def train_model(sales_df: pd.DataFrame, model_type: str = "random_forest") -> dict:
    """
    Train a model on sales_df. Returns metrics dict.
    """
    if model_type not in MODELS:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(MODELS.keys())}")

    feat_df = build_features(sales_df)
    if feat_df.empty or len(feat_df) < 2:
        raise ValueError("Not enough data to train. Need at least 2 SKUs with sales history.")

    target = compute_target(sales_df)
    feat_df = feat_df[feat_df["sku"].isin(target.index)].copy()
    feat_df["target"] = feat_df["sku"].map(target)
    feat_df.dropna(subset=["target"], inplace=True)

    X = feat_df[FEATURE_COLS].fillna(0).values
    y = feat_df["target"].values

    if len(X) < 2:
        raise ValueError("Need at least 2 samples to train.")

    # Re-create a fresh copy of the pipeline to avoid state issues
    import copy
    pipeline = copy.deepcopy(MODELS[model_type])
    pipeline.fit(X, y)

    # Cross-val metrics (if enough samples)
    if len(X) >= 5:
        cv_scores = cross_val_score(
            copy.deepcopy(MODELS[model_type]), X, y,
            cv=min(5, len(X)), scoring="neg_mean_absolute_error"
        )
        mae_cv = float(-cv_scores.mean())
    else:
        y_pred = pipeline.predict(X)
        mae_cv = float(mean_absolute_error(y, y_pred))

    y_pred_train = pipeline.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred_train)))
    mae_train = float(mean_absolute_error(y, y_pred_train))

    # Save model
    fname = f"model_{model_type}.pkl"
    fpath = os.path.join(MODEL_DIR, fname)
    joblib.dump(pipeline, fpath)

    # Save metadata
    meta = _load_metadata()
    meta[model_type] = {
        "model_type": model_type,
        "filename": fname,
        "trained_at": datetime.utcnow().isoformat(),
        "training_samples": int(len(X)),
        "metrics": {
            "mae_cv": round(mae_cv, 4),
            "mae_train": round(mae_train, 4),
            "rmse_train": round(rmse, 4),
        }
    }
    _save_metadata(meta)

    return {
        "model_type": model_type,
        "training_samples": len(X),
        "mae": round(mae_cv, 4),
        "rmse": round(rmse, 4),
    }


# ──────────────────────────────────────────────
#  Predict
# ──────────────────────────────────────────────

def predict_demand(sales_df: pd.DataFrame, inventory_df: pd.DataFrame,
                   model_type: str = "random_forest") -> list:
    """
    Returns list of dicts with: sku, name, current_stock,
    predicted_demand_7d, reorder_threshold, alert, trend_direction.
    
    Falls back to heuristic prediction if no trained model exists.
    """
    feat_df = build_features(sales_df)

    # Try loading trained model
    pipeline = _load_model(model_type)
    use_ml = (pipeline is not None) and (not feat_df.empty) and (len(feat_df) >= 1)

    results = []

    for _, row in inventory_df.iterrows():
        sku = str(row.get("sku") or row.get("SKU") or "")
        name = str(row.get("name") or row.get("Product") or sku)
        stock = float(row.get("current_stock") or row.get("Stock") or 0)
        threshold = float(row.get("reorder_threshold") or max(5, stock * 0.2))

        # Get features for this SKU
        sku_feat = feat_df[feat_df["sku"] == sku] if not feat_df.empty else pd.DataFrame()

        if use_ml and not sku_feat.empty:
            X = sku_feat[FEATURE_COLS].fillna(0).values
            predicted = max(0.0, float(pipeline.predict(X)[0]))
            trend = float(sku_feat["trend"].values[0])
            avg_daily = float(sku_feat["avg_daily_7d"].values[0])
        else:
            # Heuristic fallback: use mean of recent sales if available
            if not feat_df.empty and sku in feat_df["sku"].values:
                row_f = feat_df[feat_df["sku"] == sku].iloc[0]
                avg_daily = float(row_f["avg_daily_7d"])
                trend = float(row_f["trend"])
                predicted = avg_daily * 7
            else:
                # No data at all for this SKU — use a conservative estimate
                avg_daily = max(1.0, stock * 0.05)
                trend = 0.0
                predicted = avg_daily * 7

        predicted = round(predicted, 1)
        days_until_stockout = round(stock / avg_daily, 1) if avg_daily > 0 else 999
        alert = stock < threshold or stock < predicted

        if trend > 0.5:
            trend_dir = "rising"
        elif trend < -0.5:
            trend_dir = "falling"
        else:
            trend_dir = "stable"

        results.append({
            "sku": sku,
            "name": name,
            "current_stock": int(stock),
            "predicted_demand_7d": predicted,
            "avg_daily_sales": round(avg_daily, 2),
            "days_until_stockout": min(days_until_stockout, 999),
            "reorder_threshold": int(threshold),
            "alert": bool(alert),
            "trend": trend_dir,
        })

    # Sort: alerts first, then by days_until_stockout
    results.sort(key=lambda x: (not x["alert"], x["days_until_stockout"]))
    return results


# ──────────────────────────────────────────────
#  Sales Trend Analytics
# ──────────────────────────────────────────────

def get_sales_trends(sales_df: pd.DataFrame, days: int = 30) -> dict:
    """Returns daily totals and per-category breakdown for the last N days."""
    if sales_df.empty:
        return {"daily": [], "by_sku": []}

    sales_df = sales_df.copy()
    sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], format="mixed", dayfirst=False)
    cutoff = sales_df["sold_at"].max() - pd.Timedelta(days=days)
    recent = sales_df[sales_df["sold_at"] >= cutoff]

    daily = (
        recent.groupby(recent["sold_at"].dt.date)["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"sold_at": "date", "quantity": "total"})
    )
    daily["date"] = daily["date"].astype(str)

    by_sku = (
        recent.groupby("sku")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"quantity": "total_sold"})
    )

    return {
        "daily": daily.to_dict(orient="records"),
        "by_sku": by_sku.to_dict(orient="records"),
    }


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _load_model(model_type: str):
    fpath = os.path.join(MODEL_DIR, f"model_{model_type}.pkl")
    if os.path.exists(fpath):
        try:
            return joblib.load(fpath)
        except Exception:
            return None
    return None


def _load_metadata() -> dict:
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_metadata(meta: dict):
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)


def get_model_info() -> dict:
    return _load_metadata()

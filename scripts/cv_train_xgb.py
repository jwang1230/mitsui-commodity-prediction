# Loads processed features & labels.
# Selects a feature set (default: log returns only, can switch to “all”).
# Applies causal missing handling: forward-fill per column on train, then fill remaining NaNs with train medians; apply same medians to val.
# Trains an XGBoost regressor per target for each fold.
# Scores each fold with your rank-correlation Sharpe metric (metric.py).
# Saves per-fold scores and the overall mean/std.
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ML
import xgboost as xgb

# Local modules
# Adjust these imports if your repo layout differs
from src.tscv import PurgedTimeSeriesSplit, attach_order
from src.metrics import calculate_competition_score  # teammate's metric.py

def load_processed(processed_dir: Path):
    train_X = pd.read_csv(processed_dir / "train_features_engineered.csv")
    train_y = pd.read_csv(processed_dir / "train_labels.csv")
    # Optional: if you want to restrict to the same dates in both
    common_dates = np.intersect1d(train_X["date_id"].values, train_y["date_id"].values)
    train_X = train_X[train_X["date_id"].isin(common_dates)].copy()
    train_y = train_y[train_y["date_id"].isin(common_dates)].copy()
    train_X = train_X.sort_values("date_id").reset_index(drop=True)
    train_y = train_y.sort_values("date_id").reset_index(drop=True)
    return train_X, train_y

def select_feature_columns(train_X: pd.DataFrame, mode: str = "log_returns") -> list[str]:
    """
    mode:
      - 'log_returns': only *_log_return columns (baseline parity)
      - 'all': use every column except ['date_id']
      - 'custom': extend with your preferred rules
    """
    if mode == "log_returns":
        cols = [c for c in train_X.columns if c.endswith("_log_return")]
    elif mode == "all":
        cols = [c for c in train_X.columns if c != "date_id"]
    else:
        raise ValueError("Unknown mode; use 'log_returns' or 'all'")
    if not cols:
        raise ValueError("No feature columns found for selected mode.")
    return cols

def fit_imputer(train_block: pd.DataFrame) -> pd.Series:
    """
    Causal missing handling:
      1) forward-fill within the train block (per column)
      2) compute train medians for remaining NaNs
    Returns a Series of medians to reuse on validation.
    """
    # Step 1: ffill within train only
    train_block_ffill = train_block.copy()
    train_block_ffill[:] = train_block_ffill.ffill(axis=0)
    # Step 2: medians for remaining NaNs
    med = train_block_ffill.median(axis=0, numeric_only=True)
    # Replace any medians that remain NaN with 0.0 as a last resort
    med = med.fillna(0.0)
    return med

def apply_imputer(df_block: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    """
    Apply forward-fill on the block, then fill remaining NaNs with medians learned on train.
    """
    out = df_block.copy()
    out[:] = out.ffill(axis=0)
    # Align medians to columns
    med_aligned = med.reindex(out.columns)
    out = out.fillna(med_aligned)
    # Still any NaNs? Replace with 0 safely (should be rare).
    out = out.fillna(0.0)
    return out

def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    params: dict | None = None,
) -> tuple[xgb.XGBRegressor, np.ndarray]:
    params = params or {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.075,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "verbosity": 0,
    }
    model = xgb.XGBRegressor(**params)
    # We could add early_stopping with an inner split; here we keep it simple/fast
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return model, preds

def main(
    processed_dir: Path,
    out_dir: Path,
    n_splits: int = 3,
    val_size: int = 90,
    gap_size: int = 30,
    min_train_size: int = 600,
    step: int | None = None,
    feature_mode: str = "log_returns",
    target_limit: int | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_full, y_full = load_processed(processed_dir)
    # Attach row_id and ensure order
    X_full = attach_order(X_full, "date_id")
    y_full = attach_order(y_full, "date_id")

    # Feature columns
    feat_cols = select_feature_columns(X_full, feature_mode)

    # Targets
    target_cols = [c for c in y_full.columns if c.startswith("target_")]
    if target_limit:
        target_cols = target_cols[:target_limit]

    # Build splitter
    splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits,
        val_size=val_size,
        gap_size=gap_size,
        min_train_size=min_train_size,
        step=step,
    )

    fold_results = []
    per_target_fold_scores = []  # store fold score per target if needed

    # Iterate folds
    for fold_id, fold in enumerate(splitter.split(X_full), start=1):
        print(f"\n=== Fold {fold_id} ===")
        print(f"Train: {fold.train_range}, Gap: {fold.gap_range}, Val: {fold.val_range}")

        # Slice blocks by row index
        X_tr_raw = X_full.iloc[fold.train_idx][feat_cols].reset_index(drop=True)
        X_va_raw = X_full.iloc[fold.val_idx][feat_cols].reset_index(drop=True)
        y_tr = y_full.iloc[fold.train_idx][target_cols].reset_index(drop=True)
        y_va = y_full.iloc[fold.val_idx][target_cols].reset_index(drop=True)

        # Imputer fit on train block only (causal)
        med = fit_imputer(X_tr_raw)

        # Apply to train/val
        X_tr = apply_imputer(X_tr_raw, med)
        X_va = apply_imputer(X_va_raw, med)

        # Predict all targets for this fold
        preds_va = pd.DataFrame(index=y_va.index, columns=target_cols, dtype=float)

        # Train per target
        for tgt in tqdm(target_cols, desc=f"Fold {fold_id} training targets"):
            # Align rows where target is not null in TRAIN
            y_tr_t = y_tr[tgt]
            non_null_train = y_tr_t.notna()
            if non_null_train.sum() < 30:
                # too few samples; skip (leave predictions NaN)
                continue
            X_tr_t = X_tr.loc[non_null_train]

            # Model + predict
            model, pred = train_xgb(X_tr_t, y_tr_t[non_null_train], X_va)

            # Store predictions for val rows
            preds_va[tgt] = pred

        # Score this fold with competition metric
        # metric expects DataFrames with the same columns; intersect in the helper
        fold_score = calculate_competition_score(y_va, preds_va)
        print(f"Fold {fold_id} score (rank-corr Sharpe): {fold_score:.4f}")

        fold_results.append({"fold": fold_id, "score": float(fold_score)})

        # Optional: keep per-target Spearman? The provided metric returns a single Sharpe-like score.
        # If you want per-target simple Spearman across the val block (not the official metric), you can add it here.

        # Save per-fold predictions if desired
        preds_path = out_dir / f"preds_fold{fold_id}.csv"
        preds_out = preds_va.copy()
        preds_out.insert(0, "date_id", y_full.iloc[fold.val_idx]["date_id"].values)
        preds_out.to_csv(preds_path, index=False)

    # Aggregate and save fold results
    scores = np.array([r["score"] for r in fold_results], dtype=float)
    summary = {
        "n_splits": n_splits,
        "val_size": val_size,
        "gap_size": gap_size,
        "min_train_size": min_train_size,
        "step": step if step is not None else val_size,
        "feature_mode": feature_mode,
        "target_count": len(target_cols),
        "fold_scores": fold_results,
        "mean_score": float(scores.mean()) if len(scores) else None,
        "std_score": float(scores.std(ddof=0)) if len(scores) else None,
    }
    print("\n=== CV Summary ===")
    print(json.dumps(summary, indent=2))

    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="results/tscv_xgb")
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--val_size", type=int, default=90)
    parser.add_argument("--gap_size", type=int, default=30)
    parser.add_argument("--min_train_size", type=int, default=600)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--feature_mode", type=str, default="log_returns", choices=["log_returns", "all"])
    parser.add_argument("--target_limit", type=int, default=None, help="limit number of targets for quick runs")
    args = parser.parse_args()

    main(
        processed_dir=Path(args.processed_dir),
        out_dir=Path(args.out_dir),
        n_splits=args.n_splits,
        val_size=args.val_size,
        gap_size=args.gap_size,
        min_train_size=args.min_train_size,
        step=args.step,
        feature_mode=args.feature_mode,
        target_limit=args.target_limit,
    )

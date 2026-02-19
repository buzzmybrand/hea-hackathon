"""
preprocess.py
Imputation, scaling, splitting utilities.
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from .utils import log

def preprocess(df, feature_cols, target_col, na_thresh=0.60):
    log.info("Preprocessing: filtering, imputing, scaling ...")
    sub = df[feature_cols + [target_col]].copy()
    y = sub[target_col].values.astype(np.int8)
    X = sub.drop(columns=[target_col])
    miss = X.isnull().mean()
    keep = miss[miss <= na_thresh].index.tolist()
    X = X[keep]
    log.info(f"Kept {len(keep)}/{len(feature_cols)} features after {na_thresh:.0%} NA filter")
    X = X.apply(pd.to_numeric, errors="coerce")
    # Clip outliers per column using IQR method
    for col in X.columns:
        q1, q3 = X[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        X[col] = X[col].clip(q1 - 3*iqr, q3 + 3*iqr)
    imp = KNNImputer(n_neighbors=5, weights="distance")
    X_arr = imp.fit_transform(X.values.astype(np.float32))
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_arr).astype(np.float32)
    log.info(f"Final feature matrix shape: {X_scaled.shape}")
    return X_scaled, y, keep

def make_splits(X, y, cfg):
    """
    Return dicts: X_splits, y_splits, idx_splits
    cfg is expected to have train_frac, val_frac, test_frac, holdout_frac, random_seed
    """
    n = len(y)
    idx = np.arange(n)
    idx_dev, idx_holdout = train_test_split(idx, test_size=cfg["holdout_frac"], stratify=y, random_state=cfg["random_seed"])
    remaining = 1.0 - cfg["holdout_frac"]
    val_within  = cfg["val_frac"]  / remaining
    test_within = cfg["test_frac"] / remaining
    idx_train_val, idx_test = train_test_split(idx_dev, test_size=test_within, stratify=y[idx_dev], random_state=cfg["random_seed"])
    idx_train, idx_val = train_test_split(idx_train_val, test_size=val_within / (1 - test_within), stratify=y[idx_train_val], random_state=cfg["random_seed"])
    splits_idx = dict(train=idx_train, val=idx_val, test=idx_test, holdout=idx_holdout)
    X_splits = {k: X[v] for k, v in splits_idx.items()}
    y_splits = {k: y[v] for k, v in splits_idx.items()}
    for k in ["train","val","test","holdout"]:
        pos = y_splits[k].mean()
        log.info(f"  {k:8s} → n={len(y_splits[k]):7,} | pos_rate={pos:.3%}")
    return X_splits, y_splits, splits_idx

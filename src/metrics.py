"""
metrics.py
Evaluation metrics, threshold search, and fairness audit.
"""
import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_recall_curve, roc_auc_score, average_precision_score, precision_score, recall_score, brier_score_loss
import pandas as pd
from .utils import log

def compute_metrics(y_true, y_prob, threshold=0.5, split_name=""):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "split": split_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
    }
    log.info(f"[{split_name}] ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | F2={metrics['f2']:.4f} | Recall={metrics['recall']:.4f}")
    return metrics

def find_best_threshold(y_val, y_prob_val):
    best_t, best_f2 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.02):
        f2 = fbeta_score(y_val, (y_prob_val >= t).astype(int), beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2; best_t = t
    log.info(f"Optimal threshold (F2={best_f2:.4f}): {best_t:.2f}")
    return best_t

def fairness_audit(df_raw, holdout_idx, y_holdout, prob_holdout, threshold, protected_attrs=None):
    records = []
    df_hold = df_raw.iloc[holdout_idx].copy()
    df_hold["__prob__"] = prob_holdout
    df_hold["__y__"] = y_holdout
    df_hold["__pred__"] = (prob_holdout >= threshold).astype(int)
    for attr in (protected_attrs or []):
        col = attr.lower()
        if col not in df_hold.columns:
            continue
        for grp, gdf in df_hold.groupby(col):
            if gdf["__y__"].nunique() < 2 or len(gdf) < 30:
                continue
            row = {"attribute": col, "group": grp, "n": len(gdf), "pos_rate": gdf["__y__"].mean()}
            try:
                row["roc_auc"] = roc_auc_score(gdf["__y__"], gdf["__prob__"])
                row["f2"] = fbeta_score(gdf["__y__"], gdf["__pred__"], beta=2, zero_division=0)
                row["recall"] = recall_score(gdf["__y__"], gdf["__pred__"], zero_division=0)
            except Exception:
                pass
            records.append(row)
    if not records:
        return pd.DataFrame()
    fa = pd.DataFrame(records)
    agg_auc = roc_auc_score(y_holdout, prob_holdout)
    fa["auc_diff"] = (fa["roc_auc"] - agg_auc).abs()
    fa["flagged"] = fa["auc_diff"] > 0.05
    log.info(f"Fairness audit — flagged groups: {fa['flagged'].sum()}")
    return fa

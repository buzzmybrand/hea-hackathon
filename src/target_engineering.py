"""
target_engineering.py
Construct the health_decline target variable from HRS waves.
"""
import numpy as np
from .utils import log

# Disease flag lists should be provided by caller or imported from a constants module.
def engineer_target(df, cfg_target_col="health_decline",
                    self_rated_cols=None,
                    disease_early=None, disease_late=None):
    """
    Define target as: SRH decline >= 2 between early waves and late waves OR new chronic condition in late waves.
    Arguments:
        df: pandas DataFrame with HRS variables (lowercased).
        self_rated_cols: list of SRH columns in order (e.g., r1shlt ... r15shlt)
        disease_early: list of disease flag columns considered 'early'
        disease_late: list of disease flag columns considered 'late'
    Returns:
        df with a new column cfg_target_col
    """
    df = df.copy()
    srs = [c for c in (self_rated_cols or []) if c in df.columns]
    early_waves = srs[:5]
    late_waves = srs[-3:]
    if early_waves and late_waves:
        df["_srh_early"] = df[early_waves].apply(lambda r: pd.to_numeric(r, errors="coerce").min(), axis=1)
        df["_srh_late"]  = df[late_waves].apply(lambda r: pd.to_numeric(r, errors="coerce").max(), axis=1)
        df["_srh_decline"] = ((df["_srh_late"] - df["_srh_early"]) >= 2).astype(np.int8)
    else:
        df["_srh_decline"] = np.int8(0)
    if disease_early and disease_late:
        df["_cond_early"] = df[disease_early].apply(lambda r: pd.to_numeric(r, errors="coerce").sum(), axis=1)
        df["_cond_late"]  = df[disease_late].apply(lambda r: pd.to_numeric(r, errors="coerce").sum(), axis=1)
        df["_cond_new"]   = (df["_cond_late"] > df["_cond_early"]).astype(np.int8)
    else:
        df["_cond_new"] = np.int8(0)
    df[cfg_target_col] = ((df["_srh_decline"] == 1) | (df["_cond_new"] == 1)).astype(np.int8)
    # Drop helper columns
    leakage_cols = ["_srh_early","_srh_late","_srh_decline","_cond_early","_cond_late","_cond_new"]
    df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore", inplace=True)
    pos_rate = df[cfg_target_col].mean()
    log.info(f"Target positive rate: {pos_rate:.3%} (n={df[cfg_target_col].sum():,})")
    return df

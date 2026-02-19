"""
feature_engineering.py
Functions to compute longitudinal features (SRH trajectories, CESD, ADL/IADL, BMI, wealth, etc.)
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import log

def polyfit_trend(row):
    """Return slope of polyfit for 1-D array-like or NaN if insufficient data."""
    vals = row.dropna().values
    if len(vals) >= 2:
        return np.polyfit(np.arange(len(vals)), vals, 1)[0]
    return np.nan

def engineer_features(df):
    df = df.copy()
    pbar = tqdm(total=8, desc="Feature Engineering", unit="group")
    # SRH features: caller should provide self-rated columns named r1shlt..r15shlt as appropriate
    srh_cols = [c for c in [f"r{w}shlt" for w in range(1,16)] if c in df.columns]
    if len(srh_cols) >= 3:
        srh_mat = df[srh_cols].apply(pd.to_numeric, errors="coerce")
        df["fe_srh_mean"]    = srh_mat.mean(axis=1)
        df["fe_srh_std"]     = srh_mat.std(axis=1)
        df["fe_srh_max"]     = srh_mat.max(axis=1)
        df["fe_srh_min"]     = srh_mat.min(axis=1)
        df["fe_srh_range"]   = df["fe_srh_max"] - df["fe_srh_min"]
        df["fe_srh_trend"]   = srh_mat.apply(polyfit_trend, axis=1)
        df["fe_srh_worsened"] = (df["fe_srh_trend"] > 0).astype(np.int8)
    pbar.update(1)
    # CESD features
    cesd_cols = [c for c in [f"r{w}cesd" for w in range(1,16)] if c in df.columns]
    if len(cesd_cols) >= 3:
        cesd_mat = df[cesd_cols].apply(pd.to_numeric, errors="coerce")
        df["fe_cesd_mean"]  = cesd_mat.mean(axis=1)
        df["fe_cesd_max"]   = cesd_mat.max(axis=1)
        df["fe_cesd_trend"] = cesd_mat.apply(polyfit_trend, axis=1)
        df["fe_cesd_chronic"] = (cesd_mat >= 4).sum(axis=1)
        df["fe_cesd_spike"]   = ((cesd_mat.diff(axis=1)) >= 3).any(axis=1).astype(np.int8)
    pbar.update(1)
    # ADL / IADL
    adl_cols = [c for c in [f"r{w}adla" for w in range(1,16)] if c in df.columns]
    iadl_cols = [c for c in [f"r{w}iadlza" for w in range(1,16)] if c in df.columns]
    if adl_cols:
        adl_mat = df[adl_cols].apply(pd.to_numeric, errors="coerce")
        df["fe_adl_mean"] = adl_mat.mean(axis=1)
        df["fe_adl_trend"] = adl_mat.apply(polyfit_trend, axis=1)
    if iadl_cols:
        iadl_mat = df[iadl_cols].apply(pd.to_numeric, errors="coerce")
        df["fe_iadl_mean"] = iadl_mat.mean(axis=1)
        df["fe_iadl_trend"] = iadl_mat.apply(polyfit_trend, axis=1)
    if adl_cols and iadl_cols:
        df["fe_functional_burden"] = df.get("fe_adl_mean", 0) + df.get("fe_iadl_mean", 0)
    pbar.update(1)
    # Lifestyle
    vg_cols = [c for c in [f"r{w}vgactx" for w in range(1,16)] if c in df.columns]
    md_cols = [c for c in [f"r{w}mdactx" for w in range(1,16)] if c in df.columns]
    sm_cols = [c for c in [f"r{w}smokev" for w in range(1,16)] if c in df.columns]
    dr_cols = [c for c in [f"r{w}drink" for w in range(1,16)] if c in df.columns]
    if vg_cols:
        df["fe_vigorous_freq"] = df[vg_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if md_cols:
        df["fe_moderate_freq"] = df[md_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if vg_cols or md_cols:
        df["fe_physical_activity"] = df.get("fe_vigorous_freq", 0) * 2 + df.get("fe_moderate_freq", 0)
    if sm_cols:
        df["fe_ever_smoke"] = df[sm_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    if dr_cols:
        df["fe_ever_drink"] = df[dr_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    if sm_cols and dr_cols:
        df["fe_substance_index"] = df.get("fe_ever_smoke", 0) + df.get("fe_ever_drink", 0)
    pbar.update(1)
    # Wealth / Income
    wealth_cols = [c for c in [f"h{w}atotb" for w in range(1,16)] if c in df.columns]
    income_cols = [c for c in [f"h{w}itot" for w in range(1,16)] if c in df.columns]
    if wealth_cols:
        wealth_mat = df[wealth_cols].apply(pd.to_numeric, errors="coerce")
        df["fe_wealth_mean"] = wealth_mat.mean(axis=1)
        df["fe_wealth_trend"] = wealth_mat.apply(polyfit_trend, axis=1)
        df["fe_wealth_decline"] = (df["fe_wealth_trend"] < 0).astype(np.int8)
    if income_cols:
        income_mat = df[income_cols].apply(pd.to_numeric, errors="coerce")
        df["fe_income_mean"] = income_mat.mean(axis=1)
        df["fe_income_volatile"] = (income_mat.std(axis=1) / income_mat.mean(axis=1).abs().replace(0, np.nan))
    pbar.update(1)
    # BMI
    bmi_cols = [c for c in [f"r{w}bmi" for w in range(1,16)] if c in df.columns]
    if bmi_cols:
        bmi_mat = df[bmi_cols].apply(pd.to_numeric, errors="coerce")
        df["fe_bmi_mean"] = bmi_mat.mean(axis=1)
        df["fe_bmi_max"] = bmi_mat.max(axis=1)
        df["fe_bmi_trend"] = bmi_mat.apply(polyfit_trend, axis=1)
        df["fe_obese_ever"] = (bmi_mat >= 30).any(axis=1).astype(np.int8)
        if bmi_mat.shape[1] >= 2:
            df["fe_obese_recent"] = (bmi_mat.iloc[:, -2:] >= 30).any(axis=1).astype(np.int8)
    pbar.update(1)
    # Cross domain interactions
    if "fe_cesd_mean" in df.columns and "fe_srh_mean" in df.columns:
        df["fe_depr_x_health"] = df["fe_cesd_mean"] * df["fe_srh_mean"]
    if "fe_bmi_mean" in df.columns and "fe_cesd_mean" in df.columns:
        df["fe_bmi_x_depr"] = df["fe_bmi_mean"] * df["fe_cesd_mean"]
    if "fe_functional_burden" in df.columns and "fe_cesd_mean" in df.columns:
        df["fe_func_x_depr"] = df.get("fe_functional_burden", 0) * df["fe_cesd_mean"]
    if "fe_wealth_mean" in df.columns and "fe_srh_mean" in df.columns:
        df["fe_wealth_health"] = (df["fe_wealth_mean"] < 0).astype(float) * df["fe_srh_mean"]
    if "fe_srh_trend" in df.columns and "fe_cesd_trend" in df.columns:
        df["fe_dual_decline"] = ((df["fe_srh_trend"] > 0) & (df["fe_cesd_trend"] > 0)).astype(np.int8)
    pbar.update(1)
    # Demographics & age
    if "rabyear" in df.columns:
        df["fe_approx_age"] = 2022 - pd.to_numeric(df["rabyear"], errors="coerce")
        if "raedyrs" in df.columns:
            df["fe_education_yrs"] = pd.to_numeric(df["raedyrs"], errors="coerce")
        if "fe_srh_mean" in df.columns:
            df["fe_age_x_srh"] = df["fe_approx_age"] * df["fe_srh_mean"]
        if "fe_cesd_mean" in df.columns:
            df["fe_age_x_cesd"] = df["fe_approx_age"] * df["fe_cesd_mean"]
    if "ragender" in df.columns:
        df["fe_female"] = (pd.to_numeric(df["ragender"], errors="coerce") == 2).astype(np.int8)
    if "raracem" in df.columns:
        df["fe_race"] = pd.to_numeric(df["raracem"], errors="coerce").fillna(0).astype(np.int8)
    if "rahispan" in df.columns:
        df["fe_hispanic"] = pd.to_numeric(df["rahispan"], errors="coerce").fillna(0).astype(np.int8)
    pbar.update(1)
    pbar.close()
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    log.info(f"Engineered {len(fe_cols)} features.")
    return df

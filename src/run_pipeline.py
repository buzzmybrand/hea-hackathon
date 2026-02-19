"""
run_pipeline.py
Main script to run the full pipeline end-to-end using the modular components.
"""
import argparse
from pathlib import Path
from modules.utils import log, ensure_dirs, save_json
from modules.data_loader import load_hrs_data
from modules.target_engineering import engineer_target
from modules.feature_engineering import engineer_features
from modules.preprocess import preprocess, make_splits
from modules.models import train_lgbm, train_catboost, train_random_forest, tune_lgbm
from modules.nn_models import train_nn
from modules.ensemble import build_ensemble, test_noise_robustness
from modules.metrics import compute_metrics, find_best_threshold, fairness_audit
import numpy as np

def main(args):
    cfg = {
        "train_frac": 0.40,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "holdout_frac": 0.30,
        "random_seed": 42
    }
    ensure_dirs([args.output_dir, args.plot_dir, args.model_dir])
    df = load_hrs_data(args.input_dta)
    df = engineer_target(df, cfg_target_col="health_decline", self_rated_cols=[f"r{w}shlt" for w in range(1,16)],
                         disease_early=[f"r{w}hibp" for w in range(1,5)]+[f"r{w}diab" for w in range(1,5)],
                         disease_late=[f"r{w}hibp" for w in range(14,16)]+[f"r{w}diab" for w in range(14,16)])
    df = engineer_features(df)
    feature_cols = [c for c in df.columns if c.startswith("fe_")]
    X, y, keep = preprocess(df, feature_cols, "health_decline")
    X_splits, y_splits, idx_splits = make_splits(X, y, cfg)
    X_tr, X_val, X_test, X_hold = X_splits["train"], X_splits["val"], X_splits["test"], X_splits["holdout"]
    y_tr, y_val, y_test, y_hold = y_splits["train"], y_splits["val"], y_splits["test"], y_splits["holdout"]
    # Optional tuning
    lgb_params = tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials=10)
    lgbm = train_lgbm(X_tr, y_tr, X_val, y_val, cfg_params=lgb_params)
    cat = train_catboost(X_tr, y_tr, X_val, y_val)
    rf = train_random_forest(X_tr, y_tr)
    nn, history = train_nn(X_tr, y_tr, X_val, y_val, n_features=X_tr.shape[1])
    models = {"lgbm": (lgbm, False), "cat": (cat, False), "rf": (rf, False), "nn": (nn, True)}
    val_prob, test_prob, hold_prob, ens_weights = build_ensemble(models, X_val, y_val, X_test, y_test, X_hold, y_hold)
    best_t = find_best_threshold(y_val, val_prob)
    metrics_train = compute_metrics(y_tr, lgbm.predict_proba(X_tr)[:,1], threshold=best_t, split_name="train")
    metrics_val = compute_metrics(y_val, val_prob, threshold=best_t, split_name="val")
    metrics_test = compute_metrics(y_test, test_prob, threshold=best_t, split_name="test")
    metrics_hold = compute_metrics(y_hold, hold_prob, threshold=best_t, split_name="holdout")
    # Save metadata
    save_json({"ens_weights": ens_weights, "threshold": best_t, "features": keep}, Path(args.model_dir)/"model_metadata.json")
    # Noise robustness
    df_noise = test_noise_robustness(models, ens_weights, X_test, y_test, keep, best_t, out_dir=args.plot_dir)
    print("Completed pipeline.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dta", required=True, help="Path to HRS stata .dta file")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--plot-dir", default="./plots")
    parser.add_argument("--model-dir", default="./models")
    args = parser.parse_args()
    main(args)

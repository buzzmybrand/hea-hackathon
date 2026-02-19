"""
models.py
Training wrappers for LightGBM, CatBoost, RandomForest, and Optuna tuning.
"""
from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from .utils import log

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

def train_lgbm(X_tr, y_tr, X_val, y_val, cfg_params=None):
    if not LGBM_AVAILABLE:
        log.warning("LightGBM not available")
        return None
    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    default_params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": ratio,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }
    if cfg_params:
        default_params.update(cfg_params)
    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    log.info(f"LGBM best iteration: {model.best_iteration_}")
    return model

def train_catboost(X_tr, y_tr, X_val, y_val, cfg_params=None):
    if not CATBOOST_AVAILABLE:
        log.warning("CatBoost not available")
        return None
    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    default_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "scale_pos_weight": ratio,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "CPU",
        "thread_count": -1,
    }
    if cfg_params:
        default_params.update(cfg_params)
    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_val, y_val)
    model = CatBoostClassifier(**default_params)
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    log.info(f"CatBoost best iteration: {model.get_best_iteration()}")
    return model

def train_random_forest(X_tr, y_tr):
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_dict = {0: cw[0], 1: cw[1]}
    model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=10, class_weight=cw_dict, n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    return model

def tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials=30):
    if not (OPTUNA_AVAILABLE and LGBM_AVAILABLE):
        log.warning("Optuna or LGBM not available; skipping tuning.")
        return {}
    def objective(trial):
        p = {
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": (y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            "objective": "binary", "metric": "auc", "n_jobs": -1, "verbose": -1,
        }
        m = lgb.LGBMClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        preds = m.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"Optuna best LGBM AUC: {study.best_value:.4f}")
    return study.best_params

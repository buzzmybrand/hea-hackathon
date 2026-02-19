"""
ensemble.py
Functions to construct a weighted stacking ensemble and run noise robustness tests.
"""
from itertools import product as iprod
import numpy as np
from collections import defaultdict
from .utils import log, save_show
import matplotlib.pyplot as plt

def build_ensemble(models: dict, X_val, y_val, X_test, y_test, X_holdout, y_holdout):
    log.info("Building stacking ensemble ...")
    val_probs, test_probs, hold_probs = {}, {}, {}
    for name, (model, is_nn) in models.items():
        if model is None: continue
        if is_nn:
            import torch
            model.eval()
            with torch.no_grad():
                vp = model(torch.from_numpy(X_val).float()).numpy()
                tp = model(torch.from_numpy(X_test).float()).numpy()
                hp = model(torch.from_numpy(X_holdout).float()).numpy()
        else:
            vp = model.predict_proba(X_val)[:,1]
            tp = model.predict_proba(X_test)[:,1]
            hp = model.predict_proba(X_holdout)[:,1]
        val_probs[name] = vp
        test_probs[name] = tp
        hold_probs[name] = hp
    names = list(val_probs.keys())
    if not names:
        raise RuntimeError("No models available for ensemble.")
    best_score, best_w = 0, None
    weight_vals = np.arange(0,1.1,0.25)
    weight_grid = list(iprod(weight_vals, repeat=len(names)))
    for combo in weight_grid:
        w = np.array(combo)
        if w.sum() == 0: continue
        w = w / w.sum()
        ens = sum(w[i] * val_probs[n] for i,n in enumerate(names))
        from sklearn.metrics import average_precision_score
        sc = average_precision_score(y_val, ens)
        if sc > best_score:
            best_score = sc
            best_w = w.copy()
    if best_w is None:
        best_w = np.ones(len(names)) / len(names)
    weights = dict(zip(names, best_w))
    log.info(f"Ensemble weights (val PR-AUC={best_score:.4f}): {weights}")
    def weighted(probs_dict):
        return sum(weights[n] * probs_dict[n] for n in probs_dict.keys())
    return weighted(val_probs), weighted(test_probs), weighted(hold_probs), weights

def generate_noisy_data(X, y, feature_std, noise_frac=0.3, noise_scale=0.2, flip_prob=0.05, random_state=42):
    np.random.seed(random_state)
    X_noisy = X.copy()
    y_noisy = y.copy()
    n_samples = len(X)
    n_corrupt = int(n_samples * noise_frac)
    corrupt_idx = np.random.choice(n_samples, n_corrupt, replace=False)
    noise = np.random.normal(0, noise_scale * feature_std, size=(n_corrupt, X.shape[1]))
    X_noisy[corrupt_idx] += noise
    flip_mask = np.random.random(n_corrupt) < flip_prob
    y_noisy[corrupt_idx[flip_mask]] = 1 - y_noisy[corrupt_idx[flip_mask]]
    return X_noisy, y_noisy

def test_noise_robustness(models, ens_weights, X_test, y_test, feature_names, threshold, out_dir="./plots"):
    log.info("Running noise robustness tests...")
    feature_std = np.std(X_test, axis=0)
    noise_scales = [0.0, 0.1, 0.2, 0.3]
    flip_probs = [0.0, 0.02, 0.05, 0.1]
    rows = []
    for noise_scale in noise_scales:
        for flip_prob in flip_probs:
            X_noisy, y_noisy = generate_noisy_data(X_test, y_test, feature_std, noise_frac=0.3, noise_scale=noise_scale, flip_prob=flip_prob, random_state=42)
            probs_list = []
            weights = []
            for name, (model, is_nn) in models.items():
                if model is None or name not in ens_weights or ens_weights[name] == 0:
                    continue
                if is_nn:
                    import torch
                    model.eval()
                    with torch.no_grad():
                        p = model(torch.from_numpy(X_noisy).float()).numpy()
                else:
                    p = model.predict_proba(X_noisy)[:,1]
                probs_list.append(p)
                weights.append(ens_weights[name])
            if len(probs_list) == 0:
                continue
            ens_prob = np.average(np.column_stack(probs_list), weights=np.array(weights), axis=1)
            from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, recall_score, precision_score
            y_pred = (ens_prob >= threshold).astype(int)
            metrics = {
                "noise_scale": noise_scale,
                "flip_prob": flip_prob,
                "roc_auc": roc_auc_score(y_noisy, ens_prob),
                "pr_auc": average_precision_score(y_noisy, ens_prob),
                "f2": fbeta_score(y_noisy, y_pred, beta=2, zero_division=0),
                "recall": recall_score(y_noisy, y_pred, zero_division=0),
                "precision": precision_score(y_noisy, y_pred, zero_division=0),
            }
            rows.append(metrics)
    import pandas as pd
    df_noise = pd.DataFrame(rows)
    # Plotting summary
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    metrics_to_plot = ["roc_auc", "pr_auc", "f2", "recall"]
    for ax, metric in zip(axes.flatten(), metrics_to_plot):
        for flip in flip_probs:
            subset = df_noise[df_noise["flip_prob"] == flip]
            ax.plot(subset["noise_scale"], subset[metric], marker='o', label=f"flip={flip}")
        ax.set_xlabel("Noise Scale (feature std fraction)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs Noise")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_show(fig, "16_noise_robustness.png", out_dir=out_dir)
    return df_noise

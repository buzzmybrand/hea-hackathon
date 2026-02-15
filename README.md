# 🌅 Morning Hea: Early Health Risk Prediction

> 🏆 **HEA Hackathon** — "AI in Search of Hidden Health Signals" — Feb 14-15, 2026

## 🎯 What We Built

A machine learning pipeline that predicts **health decline** before clinical diagnosis — using only self-reported longitudinal data with **zero diagnosis leakage**.

### Results (Holdout Set)

| Metric | Score |
|--------|-------|
| **F2-Score** | 0.82 |
| **ROC-AUC** | 0.88 |
| **PR-AUC** | 0.72 |
| **Recall** | 95% |

> ✅ We catch **95% of people who will experience health decline**

---

## 👥 Team

| Name | Role |
|------|------|
| **Egor** | Product & Pitch |
| **Oluwatobi** | ML Engineering & Model Lead |
| **Masha** | Data Analysis |
| **Mohammed** | Public Health & Feature Design |

---

## 🔬 Approach

### Data
- **RAND Health & Retirement Study** (1992-2022)
- 45,000+ participants, 30+ years of longitudinal data
- 39 engineered features from health trajectories

### Key Design Decisions

| Decision | Why It Matters |
|----------|----------------|
| **Trajectories > Snapshots** | Health decline is a process — single measurements miss the trend |
| **No diagnosis leakage** | Disease flags excluded from inputs — model predicts, not memorizes |
| **Ensemble of 4 models** | LightGBM, CatBoost, RandomForest, Attention NN — weights optimized per validation |
| **Recall-optimized (F2)** | Better to over-alert than miss a sick person |
| **Fairness-tested** | Validated across gender, race & ethnicity — no disparities detected |

### Feature Groups

1. **Health trajectories** — mean, std, trend, range of self-rated health
2. **Depression signals** — CESD scores, chronic waves, spikes
3. **Functional limitations** — ADL/IADL trends
4. **Lifestyle composites** — physical activity, smoking, drinking
5. **Socioeconomic stress** — wealth/income volatility
6. **BMI dynamics** — trends, obesity flags
7. **Cross-domain interactions** — depression × health, BMI × depression

---

## 📁 Project Structure

```
├── README.md
├── notebooks/
│   └── early-health-risk-prediction-randhrs-1992-2022.ipynb  # Main pipeline
├── models/
│   ├── lgbm.pkl              # LightGBM model
│   ├── catboost.pkl          # CatBoost model
│   ├── rf.pkl                # RandomForest model
│   ├── earlyrisket.pt        # Attention NN (PyTorch)
│   └── model_meta.json       # Ensemble weights & config
├── outputs/
│   └── results_report.txt    # Evaluation summary
├── plots/                    # Visualizations (ROC, PR, SHAP, fairness)
├── docs/
│   └── nlp_strategy.md       # Voice/NLP integration roadmap
└── presentation/             # Pitch deck
```

---

## 🚀 Hackathon Compliance

- ✅ **Open Source** — 100% public code
- ✅ **Explainable** — Feature importance + SHAP visualizations
- ✅ **Fair** — Demographic parity analysis (gender, race, ethnicity)
- ✅ **No Data Leakage** — Only weak signals as features, no diagnoses

---

## 📊 How to Run

```bash
# Install dependencies
pip install tensorflow pandas numpy scikit-learn lightgbm catboost torch shap optuna imbalanced-learn

# Run the notebook
jupyter notebook notebooks/early-health-risk-prediction-randhrs-1992-2022.ipynb
```

**Note:** Requires RAND HRS dataset (`randhrs1992_2022v1.dta`) — available from [RAND HRS](https://hrsdata.isr.umich.edu/).

---

## 📜 License

MIT License — Free to use, modify, and distribute.

---

*Built with ❤️ in 24 hours at HEA Hackathon 2026*

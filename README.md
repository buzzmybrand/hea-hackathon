# Hea Hackathon: Early Health Risk Prediction

> 🏆 Hackathon: "AI in Search of Hidden Health Signals" — Feb 14-15, 2026

## What We're Building

A machine learning model that predicts **who will develop a health condition** before clinical diagnosis — using only self-reported data.

## Team

| Name | Role | GitHub |
|------|------|--------|
| Egor | Infrastructure & Coordination | [@buzzmybrand](https://github.com/buzzmybrand) |
| Oluwatobi | ML & Model Lead | [@tobimichigan](https://github.com/tobimichigan) |
| Masha | Data & Analysis Lead | [@mash1ne](https://github.com/mash1ne) |
| Mo | Medical & Feature Lead | [@mgassime](https://github.com/mgassime) |

## Approach

1. **Dataset:** RAND HRS (longitudinal health survey, 1992-2022)
2. **Target:** Predict disease onset (diabetes/heart disease/depression)
3. **Model:** XGBoost/LightGBM with SHAP explainability
4. **Key constraint:** No data leakage — only use features available BEFORE diagnosis

## Project Structure

```
├── README.md
├── AGENDA.md           # Team tasks and timeline
├── data/               # Dataset files (gitignored if large)
├── notebooks/          # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/                # Production code
├── docs/               # Documentation
│   └── nlp_strategy.md # NLP/voice extraction strategy
└── presentation/       # Final pitch materials
```

## Evaluation Criteria

**Primary Metrics (60%)**
- F2-Score (recall > precision)
- PR-AUC
- ROC-AUC

**Additional (40%)**
- No data leakage
- Real-world usability
- Cost efficiency
- Open source only
- Explainability
- Fairness (no demographic bias)

## Timeline

### Day 1 (Feb 14)
- 10:00 — Build starts
- 14:00 — Checkpoint #1
- 19:00 — Checkpoint #2

### Day 2 (Feb 15)
- 14:00 — Checkpoint #3
- 16:00 — **Submission deadline**

## Resources

- [RAND HRS Data](https://hrsdata.isr.umich.edu/data-products/rand)
- [NLSY97 Data](https://www.nlsinfo.org/investigator/pages/search?s=NLSY97)
- [PSID-SHELF Data](https://www.openicpsr.org/openicpsr/project/194322/version/V2/view)

---

Built with ❤️ for Hea Hackathon 2026

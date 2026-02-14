# 🎯 Hackathon Day 1 — Team Agenda

## What We're Building

A machine learning model that predicts **who will develop a health condition** before it happens — using only self-reported data (sleep, mood, lifestyle, etc.).

**Why this matters:** Hea wants to catch early warning signs from everyday check-ins, not medical tests. We train on historical survey data where we know what happened to people years later, then apply that to predict risk for new users.

**Our approach:**
1. Pick a target disease (diabetes, heart disease, or depression)
2. Use longitudinal survey data (RAND HRS — 30 years of health tracking)
3. Build a model that spots patterns BEFORE diagnosis
4. Make it explainable (why did we flag this person?)

---

## Team Roles

### 🔬 Mo — Medical & Feature Lead

**What you'll do:**
Define what we're predicting and which data points actually matter medically.

**Why we need this:**
The biggest risk is "data leakage" — accidentally using features that already reveal the disease (e.g., "takes insulin" = already has diabetes). Your medical background lets you catch these traps. You also know which signals are clinically meaningful vs. noise.

**Tasks:**
- Review RAND HRS variables (health, lifestyle, demographics)
- Choose target: which condition to predict? (diabetes onset recommended — clear diagnosis, good signal)
- Create feature shortlist: what inputs should the model use?
- Create leakage watchlist: what features to EXCLUDE

**Deliver by 14:00:** Target defined, feature plan, leakage risks documented

---

### 🤖 Oluwatobi — ML & Model Lead

**What you'll do:**
Build the prediction pipeline — from raw data to trained model with explainability.

**Why we need this:**
We're scored on F2 (recall matters most), PR-AUC, and ROC-AUC. You know how to optimize for these. Simple models (XGBoost/LightGBM) win hackathons because they're fast, explainable, and robust.

**Tasks:**
- Set up environment (Jupyter, dependencies)
- Design pipeline: load data → preprocess → train → evaluate
- Choose model (XGBoost recommended — handles missing data, fast iteration)
- Integrate SHAP for explainability (required by judges)

**Deliver by 14:00:** Pipeline skeleton ready, waiting for Mo's feature list

---

### 📊 Masha — Data & Analysis Lead

**What you'll do:**
Get the data ready and understand what we're working with.

**Why we need this:**
Raw data is messy. Missing values, weird distributions, class imbalance (most people are healthy). EDA finds problems before they break the model. Fairness analysis shows we're not biased — judges check this.

**Tasks:**
- Download RAND HRS dataset
- Run EDA: distributions, missing values, target balance
- Document data quality issues
- Check demographic splits (age/gender/ethnicity) for fairness baseline

**Deliver by 14:00:** Dataset loaded, EDA notebook, data quality summary

---

### 🔧 Egor — Infrastructure & Coordination

**What you'll do:**
Keep the team moving, handle technical setup, prepare the non-ML deliverables.

**Why we need this:**
We have online + offline teammates, tight timeline, and Nebius compute to configure. Someone needs to sync everything. Also: judges want our "NLP strategy" (how we'd extract features from text/voice) — this is a written deliverable, not code.

**Tasks:**
- Set up Nebius compute when credentials arrive
- Keep team synced (online ↔ offline)
- Track checkpoints (14:00, 19:00)
- Draft NLP/voice strategy section (how would we extract structured data from conversations?)
- Start pitch outline

**Deliver by 14:00:** Nebius ready (if credentials arrive), NLP strategy draft, pitch skeleton

---

## Decisions Needed Now

| Question | Options | Recommendation |
|----------|---------|----------------|
| Which dataset? | RAND HRS, NLSY97, PSID-SHELF | **RAND HRS** — cleanest, best docs |
| Which condition to predict? | Diabetes, heart disease, depression, general decline | **Mo to decide** — diabetes is common choice |
| Communication? | Slack, Discord, WhatsApp | ? |

---

## Timeline

| Time | What |
|------|------|
| 10:00–12:00 | Parallel work (see tasks above) |
| 12:00–13:00 | Lunch |
| 13:30 | Quick team sync |
| **14:00** | Checkpoint #1 |
| 14:00–19:00 | Model iteration |
| **19:00** | Checkpoint #2 |
| 21:00 | Day 1 ends |

---

## Scoring Reminder

**60% — Model Performance**
- F2-score (recall > precision — missing sick people is worse than false alarms)
- PR-AUC (handles imbalanced data)
- ROC-AUC (benchmark comparison)

**40% — Other Criteria**
- No data leakage
- Handles noisy real-world input
- Cost efficient (simple > complex)
- Open source only
- Explainable (SHAP)
- Fair (no demographic bias)

---

*Let's go 🚀*

# NLSY97 Dataset Evaluation

> Quick assessment for hackathon use — should we use this dataset?

## TL;DR Recommendation

**❌ NOT recommended as primary dataset**  
**✅ Could work as supplementary validation**

Stick with **RAND HRS** for main model. Here's why:

---

## What NLSY97 Is

| Aspect | Details |
|--------|---------|
| Population | Americans born 1980-1984 (now 41-45 years old) |
| Sample size | 8,984 individuals |
| Time span | 1997-2022 (25 years, 21 rounds) |
| Primary focus | Education, employment, life transitions |
| Health data | Secondary — varies by round, not standardized |

---

## ✅ Pros (Why It Could Work)

### 1. Long Longitudinal Tracking
25 years of data — can observe health trajectory from teens to mid-40s.

### 2. Rich Behavioral Data
- Substance use (alcohol, drugs, smoking)
- Risk behaviors
- Mental health indicators
- These are strong early predictors

### 3. Life Context
- Employment changes (stress signals)
- Family transitions (marriage, divorce, children)
- Income volatility
- These contextual factors → health outcomes

### 4. Younger Cohort = Early Signals
Could catch patterns that predict disease onset in 50s-60s.

### 5. Free & Public
No registration barriers like RAND HRS.

---

## ❌ Cons (Why It's Risky for This Hackathon)

### 1. Health Is NOT Primary Focus
> "Health items vary by round (check topical guide)"

This means:
- Variables change between survey waves
- No consistent health measures across time
- Need to dig through documentation to find usable features

**RAND HRS:** Health is THE focus. Variables are standardized.

### 2. Fewer Disease Outcomes
Age 41-45 = chronic diseases just starting:
- Diabetes onset: typically 45+
- Heart disease: typically 50+
- Cancer: typically 50+

**Problem:** Fewer "positive" cases to train on → class imbalance worse.

**RAND HRS:** Population 50+ → more disease outcomes to predict.

### 3. Target Variable Unclear
What exactly would we predict?
- Depression? (possible, but need to find PHQ/CES-D equivalent)
- Obesity trajectory? (BMI available?)
- Substance disorder? (possible)
- General health decline? (need consistent self-rated health measure)

**RAND HRS:** Clear targets — RwCONDE (chronic conditions), diabetes diagnosis, depression (CESD).

### 4. Time Cost
- NLS Investigator = custom extract tool (learning curve)
- Need to read topical guide to find health variables
- Map variables across rounds manually

**In a 2-day hackathon:** This burns hours we don't have.

### 5. Judges Expect Health Prediction
Task explicitly mentions:
> "Identify people who are likely to develop a disease"

NLSY97 is great for predicting employment, education, income — but disease prediction is weaker.

---

## Side-by-Side Comparison

| Factor | RAND HRS | NLSY97 |
|--------|----------|--------|
| Health focus | ✅ Primary | ⚠️ Secondary |
| Age range | 50+ | 40-45 |
| Disease outcomes | ✅ Many | ⚠️ Few |
| Standardized health vars | ✅ Yes (RAND cleaned) | ❌ Varies by round |
| Setup time | ~1 hour | ~3-4 hours |
| Class balance | Better | Worse |
| Self-reported health | ✅ R*SHLT every wave | ⚠️ Check availability |
| Depression measure | ✅ CESD (8-item) | ⚠️ Varies |

---

## When NLSY97 Would Make Sense

1. **If we had more time** — NLSY97 could find early-life predictors of later disease
2. **For validation** — train on RAND HRS, test concept on NLSY97
3. **Mental health focus** — if predicting depression/anxiety trajectory
4. **Behavioral patterns** — substance use → health outcomes

---

## Verdict

### For this hackathon: Use RAND HRS

**Reasons:**
1. Health is primary focus = less data wrangling
2. Clear target variables = faster iteration
3. More disease outcomes = better model training
4. Pre-cleaned RAND version = consistent variables
5. Better documentation for health-specific analysis

### NLSY97 as stretch goal:
If we finish main model early, could validate on NLSY97 to show generalization across age groups.

---

## Quick Decision Matrix

| Question | RAND HRS | NLSY97 |
|----------|----------|--------|
| Can we predict disease? | ✅ Yes | ⚠️ Limited |
| Time to first model? | 2-3 hours | 5-6 hours |
| Risk of wasted effort? | Low | High |
| Judge-friendly? | ✅ Yes | ⚠️ Requires explanation |

---

*Recommendation: Stay with RAND HRS. Mention NLSY97 in pitch as "future work" for younger populations.*

# Evidence-Thresholded Model Revision (ETMR)

## A Decision Framework for Reliable Scientific AI

Evidence-Thresholded Model Revision (ETMR) is a research framework for investigating whether an AI system can determine **when available evidence is sufficient to justify changing an underlying scientific model**.

Rather than optimizing prediction error alone, ETMR treats model revision as a scientific decision problem:

> **When should an AI system KEEP the existing model, RECALIBRATE its parameters, REVISE its structure, or ABSTAIN because the available evidence is insufficient?**

The framework is currently developed and evaluated in the context of an **irrigation Digital Twin**, using environmental observations and controlled model-adequacy experiments.

---

## Research Motivation

Machine learning systems can achieve excellent predictive accuracy while still failing to determine whether a scientific model itself is inadequate.

A discrepancy between model predictions and observations may arise from:

1. Measurement noise
2. Parameter uncertainty
3. Structural model inadequacy
4. Insufficient or unreliable observations

These cases require fundamentally different responses.

For example:

- Noise should generally **not trigger model modification**.
- Parameter error may require **recalibration**.
- Structural failure may justify **model revision**.
- Insufficient evidence should cause the system to **abstain rather than overclaim**.

Most conventional ML pipelines optimize prediction metrics such as MAE, RMSE, or R² without explicitly addressing this distinction.

ETMR investigates whether an AI system can learn this distinction from evidence.

---

# Core Research Question

> **Can an AI system determine when observed evidence is sufficient to justify structural scientific model revision, while avoiding unnecessary revisions caused by noise, parameter uncertainty, or insufficient evidence?**

---

# ETMR Decision Space

The framework produces four possible decisions:

| Decision | Meaning |
|---|---|
| **KEEP** | The existing scientific model remains adequate; observed discrepancies are consistent with noise or normal variation. |
| **RECALIBRATE** | The model structure is considered adequate, but its parameters require adjustment. |
| **REVISE** | Evidence indicates that the existing model structure is inadequate and structural modification is justified. |
| **ABSTAIN** | Available evidence is insufficient or unreliable to justify a confident model decision. |

The key principle is:

> **Prediction error alone should not determine whether a scientific model is changed.**

---

# Research Architecture

The current ETMR research pipeline consists of controlled stages:

1. **Scientific Model + Real-Data Benchmark**
2. **Controlled Model-Adequacy Benchmark**
3. **Evidence Generation**
4. **Evidence Characterization**
5. **Provenance-Conditioned Decision**
6. **Controlled Validation**
7. **Generalization to Unseen Cases**

The benchmark is designed so that the system can be evaluated against cases with known hidden ground truth.

---

# Scientific Setting

The current experimental domain is an **irrigation Digital Twin**.

The real-world benchmark contains environmental and soil-related observations including variables such as:

- Temperature
- Rainfall
- Solar radiation
- Surface soil moisture
- Root-zone soil moisture
- Surface wetness
- NDVI observations
- Observation/interpolation indicators
- Valid pixel fraction
- Number of available images
- Temporal lag features

The dataset currently contains:

- **2,401 observations**
- **21 variables**
- Training period: **2018-06-01 → 2023-12-31**
- Test period: **2024-01-01 → 2024-12-26**

---

# Baseline Model

The initial real-data benchmark establishes a high-performing baseline before introducing controlled model inadequacy.

Current baseline performance:

- **MAE:** 0.000054
- **RMSE:** 0.000096
- **R²:** 0.999984

The purpose of this baseline is not simply to maximize predictive performance.

It establishes a reference model against which controlled discrepancies and evidence patterns can be studied.

---

# Evidence Characterization

ETMR examines multiple characteristics of the model-observation discrepancy rather than relying solely on aggregate error.

Current evidence features include:

- Mean Absolute Error
- Residual standard deviation
- Temporal autocorrelation
- Rainfall correlation
- Temperature correlation
- 7-day rainfall correlation
- NDVI relationship
- Observation provenance
- Data completeness
- Valid pixel fraction
- Number of available images

Example baseline evidence:

```text
MAE                  = 0.0000536399
Residual Std         = 0.0000963595
Lag-1 Autocorrelation = 0.176229
Rainfall Correlation = 0.013040
Temperature Corr.    = -0.066404
Rainfall 7D Corr.    = -0.068416
NDVI Correlation     = -0.055358

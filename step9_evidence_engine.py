import pandas as pd
import numpy as np

print("=" * 70)
print("ETMR — STEP 9: EVIDENCE SUFFICIENCY ENGINE")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 2. CREATE RESIDUAL FROM BASELINE
# ============================================================

# Load baseline predictions
pred = pd.read_csv("step3_predictions.csv")

pred["date"] = pd.to_datetime(pred["date"])

df = df.merge(
    pred[["date", "prediction"]],
    on="date",
    how="left"
)

df["residual"] = (
    df["sm_surface"] - df["prediction"]
)


# ============================================================
# 3. EVIDENCE COMPONENT 1 — ERROR MAGNITUDE
# ============================================================

mae = df["residual"].abs().mean()

error_score = min(
    mae / 0.002,
    1.0
)


# ============================================================
# 4. EVIDENCE COMPONENT 2 — TEMPORAL PERSISTENCE
# ============================================================

residual = df["residual"]

temporal_corr = residual.autocorr(
    lag=1
)

temporal_corr = abs(
    temporal_corr
)

temporal_score = min(
    temporal_corr / 0.5,
    1.0
)


# ============================================================
# 5. EVIDENCE COMPONENT 3 — CONTEXT DEPENDENCE
# ============================================================

context_correlations = []

for column in [
    "rainfall",
    "temperature",
    "solar_radiation",
    "sm_surface_wetness",
    "rainfall_7d_sum",
    "ndvi"
]:

    correlation = abs(
        residual.corr(df[column])
    )

    if not np.isnan(correlation):
        context_correlations.append(
            correlation
        )

if context_correlations:

    context_score = max(
        context_correlations
    )

else:

    context_score = 0.0


# ============================================================
# 6. EVIDENCE COMPONENT 4 — PROVENANCE QUALITY
# ============================================================

# Observation quality components

observed_fraction = (
    df["is_observed"].mean()
)

mean_pixel_fraction = (
    df["valid_pixel_fraction"].mean()
)

mean_images = (
    df["n_images"].mean()
)

# Normalize approximately to [0,1]

provenance_score = (
    0.5 * observed_fraction
    +
    0.3 * mean_pixel_fraction
    +
    0.2 * min(mean_images / 10, 1.0)
)


# ============================================================
# 7. COMBINE EVIDENCE
# ============================================================

evidence_score = (

    0.25 * error_score
    +
    0.20 * temporal_score
    +
    0.30 * context_score
    +
    0.25 * provenance_score

)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print()
print("EVIDENCE COMPONENTS")
print("-" * 50)

print(
    f"Error magnitude score : {error_score:.6f}"
)

print(
    f"Temporal persistence  : {temporal_score:.6f}"
)

print(
    f"Context dependence    : {context_score:.6f}"
)

print(
    f"Provenance quality    : {provenance_score:.6f}"
)

print()
print(
    f"OVERALL EVIDENCE SCORE : {evidence_score:.6f}"
)


# ============================================================
# 9. SAVE
# ============================================================

results = pd.DataFrame({

    "error_score": [error_score],

    "temporal_score": [temporal_score],

    "context_score": [context_score],

    "provenance_score": [provenance_score],

    "evidence_score": [evidence_score]

})

results.to_csv(
    "step9_evidence_score.csv",
    index=False
)


print()
print("=" * 70)
print("Saved: step9_evidence_score.csv")
print("=" * 70)
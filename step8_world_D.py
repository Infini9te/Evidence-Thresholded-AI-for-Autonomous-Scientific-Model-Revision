import pandas as pd
import numpy as np

print("=" * 70)
print("ETMR — CONTROLLED WORLD D: INSUFFICIENT EVIDENCE")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD ORIGINAL DATA
# ------------------------------------------------------------

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ------------------------------------------------------------
# 2. ORIGINAL SOIL MOISTURE
# ------------------------------------------------------------

true_sm = df["sm_surface"].copy()

# ------------------------------------------------------------
# 3. CREATE AN APPARENT STRUCTURAL DISCREPANCY
# ------------------------------------------------------------
# We deliberately create a rainfall-dependent discrepancy,
# similar to World C.

rainfall = df["rainfall"].fillna(0)

apparent_effect = (
    0.015 * np.sqrt(np.maximum(rainfall, 0))
)

df["sm_surface_uncertain"] = (
    true_sm + apparent_effect
)

df["residual_uncertain"] = (
    df["sm_surface_uncertain"] - true_sm
)

# ------------------------------------------------------------
# 4. DEGRADE EVIDENCE QUALITY
# ------------------------------------------------------------
# Simulate a situation where most evidence comes from
# interpolation/climatology rather than direct observation.

df["evidence_quality"] = "LOW"

df["fill_method_original"] = df["fill_method"]

df["fill_method"] = "climatology"

df["valid_pixel_fraction"] = 0.20

df["n_images"] = 1

df["is_observed"] = 0

df["is_interpolated"] = 1

# ------------------------------------------------------------
# 5. GROUND TRUTH
# ------------------------------------------------------------

df["ground_truth_decision"] = "ABSTAIN"

df["failure_type"] = "insufficient_evidence"

# ------------------------------------------------------------
# 6. EVIDENCE STATISTICS
# ------------------------------------------------------------

residual = df["residual_uncertain"]

rainfall_corr = residual.corr(rainfall)

print()
print("Apparent structural effect created.")

print()
print("Residual mean   :", residual.mean())
print("Residual std    :", residual.std())
print("Residual MAE    :", residual.abs().mean())

print()
print("Residual ↔ rainfall correlation :",
      rainfall_corr)

print()
print("Evidence quality:")
print("Fill method          :", df["fill_method"].iloc[0])
print("Valid pixel fraction :", df["valid_pixel_fraction"].iloc[0])
print("Number of images     :", df["n_images"].iloc[0])

print()
print("Ground-truth decision:")
print("ABSTAIN")

# ------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------

df.to_csv(
    "step8_world_D_insufficient_evidence.csv",
    index=False
)

print()
print("=" * 70)
print("Saved: step8_world_D_insufficient_evidence.csv")
print("=" * 70)
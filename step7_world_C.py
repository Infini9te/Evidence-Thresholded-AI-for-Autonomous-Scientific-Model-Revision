import pandas as pd
import numpy as np

print("=" * 70)
print("ETMR — CONTROLLED WORLD C: STRUCTURAL ERROR")
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
# 3. CREATE A MISSING PHYSICAL MECHANISM
# ------------------------------------------------------------
# We introduce an additional nonlinear rainfall response.
#
# The current model does not explicitly contain this mechanism.
# Large rainfall events therefore create systematic structural
# discrepancies.

rainfall = df["rainfall"].fillna(0)

rainfall_scale = 0.015

missing_mechanism = (
    rainfall_scale * np.sqrt(np.maximum(rainfall, 0))
)

# ------------------------------------------------------------
# 4. CREATE STRUCTURALLY DIFFERENT "REAL WORLD"
# ------------------------------------------------------------

df["sm_surface_structural"] = (
    true_sm + missing_mechanism
)

# ------------------------------------------------------------
# 5. RESIDUAL CAUSED BY STRUCTURAL ERROR
# ------------------------------------------------------------

df["residual_structural_error"] = (
    df["sm_surface_structural"] - true_sm
)

# ------------------------------------------------------------
# 6. GROUND TRUTH
# ------------------------------------------------------------

df["ground_truth_decision"] = "REVISE"

df["failure_type"] = "missing_rainfall_mechanism"

df["rainfall_mechanism_scale"] = rainfall_scale

# ------------------------------------------------------------
# 7. EVIDENCE
# ------------------------------------------------------------

residual = df["residual_structural_error"]

rainfall_corr = residual.corr(rainfall)

print()
print("Missing mechanism:")
print("Nonlinear rainfall response")

print()
print("Mechanism scale :", rainfall_scale)

print()
print("Residual mean   :", residual.mean())
print("Residual std    :", residual.std())
print("Residual MAE    :", residual.abs().mean())

print()
print("Residual ↔ rainfall correlation :",
      rainfall_corr)

print()
print("Rainfall statistics:")
print("Mean :", rainfall.mean())
print("Max  :", rainfall.max())

print()
print("Ground-truth decision:")
print("REVISE")

# ------------------------------------------------------------
# 8. SAVE
# ------------------------------------------------------------

df.to_csv(
    "step7_world_C_structural_error.csv",
    index=False
)

print()
print("=" * 70)
print("Saved: step7_world_C_structural_error.csv")
print("=" * 70)
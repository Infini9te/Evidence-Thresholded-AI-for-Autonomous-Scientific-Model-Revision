import pandas as pd
import numpy as np

print("=" * 70)
print("ETMR — CONTROLLED WORLD B: PARAMETER ERROR")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD ORIGINAL DATA
# ------------------------------------------------------------

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ------------------------------------------------------------
# 2. ORIGINAL TARGET
# ------------------------------------------------------------

true_sm = df["sm_surface"].copy()

# ------------------------------------------------------------
# 3. CREATE PARAMETER ERROR
# ------------------------------------------------------------
# We simulate a model whose structure is correct,
# but whose response magnitude is incorrectly calibrated.

parameter_scale = 0.92

model_prediction = (
    true_sm.mean()
    + parameter_scale * (true_sm - true_sm.mean())
)

df["sm_surface_parameter_error"] = model_prediction

# ------------------------------------------------------------
# 4. CALCULATE PARAMETER-ERROR RESIDUAL
# ------------------------------------------------------------

df["residual_parameter_error"] = (
    true_sm - model_prediction
)

# ------------------------------------------------------------
# 5. RECORD GROUND TRUTH
# ------------------------------------------------------------

df["ground_truth_decision"] = "RECALIBRATE"

df["failure_type"] = "parameter_error"

df["parameter_scale"] = parameter_scale

# ------------------------------------------------------------
# 6. REPORT
# ------------------------------------------------------------

print()
print("Parameter scale :", parameter_scale)

print()
print("Original SM mean :", true_sm.mean())
print("Model mean      :", model_prediction.mean())

print()
print("Residual mean    :",
      df["residual_parameter_error"].mean())

print("Residual std     :",
      df["residual_parameter_error"].std())

print("Residual MAE     :",
      df["residual_parameter_error"].abs().mean())

print()
print("Ground-truth decision:")
print("RECALIBRATE")

# ------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------

df.to_csv(
    "step6_world_B_parameter_error.csv",
    index=False
)

print()
print("=" * 70)
print("Saved: step6_world_B_parameter_error.csv")
print("=" * 70)
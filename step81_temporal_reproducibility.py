import numpy as np
import pandas as pd

print("=" * 80)
print("ETMR — STEP 81: TEMPORAL REPRODUCIBILITY GATE")
print("=" * 80)

# ============================================================
# FROZEN PARAMETERS
# ============================================================

QUALITY_THRESHOLD = 0.80
MIN_SUPPORTED_PERIODS = 2

print("\nFrozen temporal rule:")
print(f"Minimum supported periods for REVISE = {MIN_SUPPORTED_PERIODS}")
print(f"Quality threshold                  = {QUALITY_THRESHOLD:.2f}")

# ============================================================
# MULTI-PERIOD BOOTSTRAP RESULTS FROM STEP 63
#
# Positive RMSE gain means the counterfactual improved RMSE.
# Structural residual reduction is treated separately.
# ============================================================

periods = [
    {
        "year": 2022,
        "quality": 1.00,
        "parameter_rmse_gain": -0.00003893,
        "parameter_p": 0.0195,
        "structural_rmse_gain": 0.00001186,
        "structural_p": 0.7360,
        "structural_reduction": 0.00063001,
        "structural_reduction_p": 0.6110,
    },
    {
        "year": 2023,
        "quality": 1.00,
        "parameter_rmse_gain": -0.00001669,
        "parameter_p": 0.2190,
        "structural_rmse_gain": -0.00000687,
        "structural_p": 0.3370,
        "structural_reduction": -0.00059453,
        "structural_reduction_p": 0.4115,
    },
    {
        "year": 2024,
        "quality": 1.00,
        "parameter_rmse_gain": 0.00005216,
        "parameter_p": 0.9805,
        "structural_rmse_gain": 0.00003028,
        "structural_p": 0.9650,
        "structural_reduction": -0.00312603,
        "structural_reduction_p": 0.1360,
    },
]

df = pd.DataFrame(periods)

# ============================================================
# PERIOD-LEVEL EVIDENCE
# ============================================================

# A period supports structural revision only when BOTH:
#
# 1. structural RMSE gain is positive
# 2. structural residual reduction is positive
#
# This is deliberately conservative.
#
# We do not treat a single positive metric as sufficient.

df["structural_supported"] = (
    (df["structural_rmse_gain"] > 0) &
    (df["structural_reduction"] > 0) &
    (df["quality"] >= QUALITY_THRESHOLD)
)

df["parameter_supported"] = (
    (df["parameter_rmse_gain"] > 0) &
    (df["parameter_p"] > 0.95) &
    (df["quality"] >= QUALITY_THRESHOLD)
)

print("\n" + "=" * 80)
print("PERIOD-LEVEL EVIDENCE")
print("=" * 80)

print(
    df[
        [
            "year",
            "parameter_rmse_gain",
            "parameter_p",
            "structural_rmse_gain",
            "structural_p",
            "structural_reduction",
            "structural_reduction_p",
            "structural_supported",
            "parameter_supported",
        ]
    ].to_string(index=False)
)

# ============================================================
# TEMPORAL REPRODUCIBILITY
# ============================================================

structural_supported_periods = int(
    df["structural_supported"].sum()
)

parameter_supported_periods = int(
    df["parameter_supported"].sum()
)

print("\n" + "=" * 80)
print("TEMPORAL REPRODUCIBILITY")
print("=" * 80)

print(
    f"Structural-supported periods = "
    f"{structural_supported_periods}/{len(df)}"
)

print(
    f"Parameter-supported periods = "
    f"{parameter_supported_periods}/{len(df)}"
)

# ============================================================
# FINAL TEMPORAL DECISION
# ============================================================

if structural_supported_periods >= MIN_SUPPORTED_PERIODS:
    temporal_decision = "REVISE"
elif parameter_supported_periods >= MIN_SUPPORTED_PERIODS:
    temporal_decision = "RECALIBRATE"
else:
    temporal_decision = "KEEP"

print("\n" + "=" * 80)
print("FINAL TEMPORAL ETMR DECISION")
print("=" * 80)

print(f"Decision = {temporal_decision}")

# ============================================================
# INTERPRETATION
# ============================================================

if temporal_decision == "REVISE":
    interpretation = (
        "Structural evidence is reproducible across multiple "
        "independent temporal periods."
    )

elif temporal_decision == "RECALIBRATE":
    interpretation = (
        "Parameter correction has reproducible support across "
        "multiple independent temporal periods, while structural "
        "revision does not."
    )

else:
    interpretation = (
        "Neither structural revision nor parameter recalibration "
        "has sufficient reproducible temporal support. "
        "The scientifically conservative decision is KEEP."
    )

print("\nInterpretation:")
print(interpretation)

# ============================================================
# SAVE
# ============================================================

output_file = "step81_temporal_reproducibility_results.csv"

df.to_csv(output_file, index=False)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 80)
print("STEP 81 COMPLETE")
print("=" * 80)
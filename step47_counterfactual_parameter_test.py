import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 47: COUNTERFACTUAL PARAMETER ADEQUACY TEST")
print("=" * 80)

df = pd.read_csv("step42_standardized_evidence.csv")

# Only parameter-error and structural-error cases
df = df[
    df["Expected"].isin(["RECALIBRATE", "REVISE"])
].copy()

# ---------------------------------------------------------
# PARAMETER EXPLANATION SCORE
# ---------------------------------------------------------
#
# A parameter error should produce a strong persistent
# discrepancy that is coherent with the existing model.
#
# Structural evidence should require an independent
# mechanism signature beyond persistence.
#
# We therefore estimate how much of the discrepancy
# can be explained by persistence alone.
# ---------------------------------------------------------

df["Parameter_Explanation"] = (
    df["Temporal"] *
    (1 - df["Lagged_Rainfall"]) *
    (1 - df["Interaction"])
)

# ---------------------------------------------------------
# INDEPENDENT STRUCTURAL EVIDENCE
# ---------------------------------------------------------

df["Independent_Structural_Evidence"] = (
    df[
        [
            "Context",
            "Lagged_Rainfall",
            "Interaction"
        ]
    ].max(axis=1)
)

# ---------------------------------------------------------
# DISCRIMINATION MARGIN
# ---------------------------------------------------------

df["Structural_Excess"] = (
    df["Independent_Structural_Evidence"]
    - df["Parameter_Explanation"]
)

# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("COUNTERFACTUAL PARAMETER ADEQUACY")
print("=" * 80)

cols = [
    "World",
    "Expected",
    "Temporal",
    "Context",
    "Lagged_Rainfall",
    "Interaction",
    "Parameter_Explanation",
    "Independent_Structural_Evidence",
    "Structural_Excess"
]

print(
    df[cols].to_string(
        index=False,
        formatters={
            "Temporal": "{:.3f}".format,
            "Context": "{:.3f}".format,
            "Lagged_Rainfall": "{:.3f}".format,
            "Interaction": "{:.3f}".format,
            "Parameter_Explanation": "{:.3f}".format,
            "Independent_Structural_Evidence": "{:.3f}".format,
            "Structural_Excess": "{:.3f}".format
        }
    )
)

# ---------------------------------------------------------
# GROUP SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("GROUP SUMMARY")
print("=" * 80)

summary = df.groupby("Expected")[
    [
        "Parameter_Explanation",
        "Independent_Structural_Evidence",
        "Structural_Excess"
    ]
].agg(["mean", "std", "min", "max"])

print(summary.to_string())

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv(
    "step47_counterfactual_parameter_test.csv",
    index=False
)

print("\nSaved:")
print("step47_counterfactual_parameter_test.csv")

print("\n" + "=" * 80)
print("STEP 47 COMPLETE")
print("=" * 80)
import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 48: ACTUAL PARAMETER RECALIBRATION COUNTERFACTUAL")
print("=" * 80)

df = pd.read_csv("step42_standardized_evidence.csv")

# ---------------------------------------------------------
# PARAMETER + STRUCTURAL CASES
# ---------------------------------------------------------

df = df[
    df["Expected"].isin(["RECALIBRATE", "REVISE"])
].copy()

# ---------------------------------------------------------
# IMPORTANT:
# For this controlled benchmark, the discrepancy is
# represented through MAE and persistence/context evidence.
#
# We construct a normalized parameter-adjustment potential
# from the observed discrepancy pattern.
#
# Persistent discrepancy = candidate parameter error.
# Independent temporal/context mechanism = candidate
# structural error.
# ---------------------------------------------------------

# Parameter adjustment potential
df["Parameter_Adjustment_Potential"] = (
    df["Temporal"] *
    (1 - df["Lagged_Rainfall"]) *
    (1 - df["Interaction"])
)

# Residual unexplained after parameter adjustment
df["Residual_After_Parameter"] = (
    1 - df["Parameter_Adjustment_Potential"]
)

# Structural evidence remaining after parameter explanation
df["Structural_Evidence_After_Parameter"] = (
    df["Context"] *
    df["Residual_After_Parameter"]
)

# ---------------------------------------------------------
# COUNTERFACTUAL GAIN
# ---------------------------------------------------------

df["Parameter_Gain"] = (
    df["Parameter_Adjustment_Potential"]
)

df["Structural_Gain"] = (
    df["Structural_Evidence_After_Parameter"]
)

# ---------------------------------------------------------
# DECISION
# ---------------------------------------------------------

def counterfactual_decision(row):

    parameter_gain = row["Parameter_Gain"]
    structural_gain = row["Structural_Gain"]

    if parameter_gain >= 0.70 and parameter_gain > structural_gain:
        return "RECALIBRATE"

    if structural_gain >= 0.30:
        return "REVISE"

    return "AMBIGUOUS"


df["Counterfactual_Decision"] = df.apply(
    counterfactual_decision,
    axis=1
)

df["Correct"] = (
    (
        (df["Expected"] == "RECALIBRATE") &
        (df["Counterfactual_Decision"] == "RECALIBRATE")
    )
    |
    (
        (df["Expected"] == "REVISE") &
        (df["Counterfactual_Decision"] == "REVISE")
    )
)

# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("COUNTERFACTUAL RESULTS")
print("=" * 80)

cols = [
    "World",
    "Expected",
    "Temporal",
    "Context",
    "Lagged_Rainfall",
    "Interaction",
    "Parameter_Gain",
    "Structural_Gain",
    "Counterfactual_Decision",
    "Correct"
]

print(
    df[cols].to_string(
        index=False,
        formatters={
            "Temporal": "{:.3f}".format,
            "Context": "{:.3f}".format,
            "Lagged_Rainfall": "{:.3f}".format,
            "Interaction": "{:.3f}".format,
            "Parameter_Gain": "{:.3f}".format,
            "Structural_Gain": "{:.3f}".format
        }
    )
)

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

accuracy = df["Correct"].mean()

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print("Correct:", df["Correct"].sum())
print("Total  :", len(df))
print("Accuracy:", f"{accuracy:.3f}")

print("\nConfusion matrix:")

print(
    pd.crosstab(
        df["Expected"],
        df["Counterfactual_Decision"]
    )
)

# ---------------------------------------------------------
# GROUP STATISTICS
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("GROUP COUNTERFACTUAL GAIN")
print("=" * 80)

print(
    df.groupby("Expected")[
        [
            "Parameter_Gain",
            "Structural_Gain"
        ]
    ].agg(["mean", "std", "min", "max"]).to_string()
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv(
    "step48_parameter_counterfactual.csv",
    index=False
)

print("\nSaved:")
print("step48_parameter_counterfactual.csv")

print("\n" + "=" * 80)
print("STEP 48 COMPLETE")
print("=" * 80)
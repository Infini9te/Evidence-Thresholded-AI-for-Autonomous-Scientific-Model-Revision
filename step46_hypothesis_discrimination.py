import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 46: PARAMETER vs STRUCTURAL HYPOTHESIS DISCRIMINATION")
print("=" * 80)

df = pd.read_csv("step42_standardized_evidence.csv")

# ---------------------------------------------------------
# SELECT PARAMETER + STRUCTURAL CASES
# ---------------------------------------------------------

subset = df[
    df["Expected"].isin(["RECALIBRATE", "REVISE"])
].copy()

print("\nSelected cases:", len(subset))

# ---------------------------------------------------------
# EVIDENCE SIGNATURES
# ---------------------------------------------------------

features = [
    "MAE",
    "Temporal",
    "Context",
    "Lagged_Rainfall",
    "Interaction",
    "Q_evidence"
]

print("\n" + "=" * 80)
print("GROUP STATISTICS")
print("=" * 80)

summary = subset.groupby("Expected")[features].agg(
    ["mean", "std", "min", "max"]
)

print(summary.to_string())

# ---------------------------------------------------------
# SIMPLE HYPOTHESIS SCORES
# ---------------------------------------------------------

def parameter_support(row):
    persistence = row["Temporal"]

    # Parameter errors should produce persistent
    # discrepancies without strong independent
    # structural signatures.
    independent_structure = max(
        row["Lagged_Rainfall"],
        row["Interaction"]
    )

    return persistence * (1 - independent_structure)


def structural_support(row):
    return max(
        row["Context"],
        row["Lagged_Rainfall"],
        row["Interaction"]
    )


subset["Parameter_Support"] = subset.apply(
    parameter_support,
    axis=1
)

subset["Structural_Support"] = subset.apply(
    structural_support,
    axis=1
)

subset["Structural_minus_Parameter"] = (
    subset["Structural_Support"]
    - subset["Parameter_Support"]
)

# ---------------------------------------------------------
# DISPLAY EACH CASE
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CASE-BY-CASE HYPOTHESIS EVIDENCE")
print("=" * 80)

display_cols = [
    "World",
    "Expected",
    "MAE",
    "Temporal",
    "Context",
    "Lagged_Rainfall",
    "Interaction",
    "Parameter_Support",
    "Structural_Support",
    "Structural_minus_Parameter"
]

print(
    subset[display_cols].to_string(
        index=False,
        formatters={
            "MAE": "{:.6f}".format,
            "Temporal": "{:.3f}".format,
            "Context": "{:.3f}".format,
            "Lagged_Rainfall": "{:.3f}".format,
            "Interaction": "{:.3f}".format,
            "Parameter_Support": "{:.3f}".format,
            "Structural_Support": "{:.3f}".format,
            "Structural_minus_Parameter": "{:.3f}".format
        }
    )
)

# ---------------------------------------------------------
# SIMPLE HYPOTHESIS DECISION
# ---------------------------------------------------------

def hypothesis_decision(row):

    p = row["Parameter_Support"]
    s = row["Structural_Support"]

    if p > s:
        return "PARAMETER"

    elif s > p:
        return "STRUCTURAL"

    else:
        return "AMBIGUOUS"


subset["Hypothesis_Decision"] = subset.apply(
    hypothesis_decision,
    axis=1
)

subset["Hypothesis_Correct"] = (
    (
        (subset["Expected"] == "RECALIBRATE") &
        (subset["Hypothesis_Decision"] == "PARAMETER")
    )
    |
    (
        (subset["Expected"] == "REVISE") &
        (subset["Hypothesis_Decision"] == "STRUCTURAL")
    )
)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("HYPOTHESIS DISCRIMINATION RESULTS")
print("=" * 80)

print(
    subset[
        [
            "World",
            "Expected",
            "Hypothesis_Decision",
            "Hypothesis_Correct"
        ]
    ].to_string(index=False)
)

accuracy = subset["Hypothesis_Correct"].mean()

print("\nCorrect:", subset["Hypothesis_Correct"].sum())
print("Total  :", len(subset))
print("Accuracy:", f"{accuracy:.3f}")

print("\nConfusion matrix:")

print(
    pd.crosstab(
        subset["Expected"],
        subset["Hypothesis_Decision"]
    )
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

subset.to_csv(
    "step46_hypothesis_discrimination.csv",
    index=False
)

print("\nSaved:")
print("step46_hypothesis_discrimination.csv")

print("\n" + "=" * 80)
print("STEP 46 COMPLETE")
print("=" * 80)
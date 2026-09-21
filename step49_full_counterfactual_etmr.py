import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 49: FULL COUNTERFACTUAL DECISION ENGINE")
print("=" * 80)

df = pd.read_csv("step42_standardized_evidence.csv")

# ---------------------------------------------------------
# DECISION ENGINE
# ---------------------------------------------------------

QUALITY_THRESHOLD = 0.80

def etmr_counterfactual(row):

    q = row["Q_evidence"]
    persistence = row["Temporal"]
    context = row["Context"]
    lagged = row["Lagged_Rainfall"]
    interaction = row["Interaction"]

    # -----------------------------------------------------
    # 1. EVIDENCE SUFFICIENCY
    # -----------------------------------------------------

    if q < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # -----------------------------------------------------
    # 2. PARAMETER COUNTERFACTUAL
    # -----------------------------------------------------

    parameter_gain = (
        persistence *
        (1 - lagged) *
        (1 - interaction)
    )

    structural_gain = (
        context *
        (1 - parameter_gain)
    )

    # Independent lagged / interaction evidence
    if lagged >= 0.70:
        structural_gain = max(
            structural_gain,
            lagged
        )

    if interaction >= 0.60:
        structural_gain = max(
            structural_gain,
            interaction
        )

    # -----------------------------------------------------
    # 3. PARAMETER-EXPLAINABLE DISCREPANCY
    # -----------------------------------------------------

    if (
        parameter_gain >= 0.70
        and structural_gain < 0.30
    ):
        return "RECALIBRATE"

    # -----------------------------------------------------
    # 4. STRUCTURAL REVISION
    # -----------------------------------------------------

    if structural_gain >= 0.30:
        return "REVISE"

    # -----------------------------------------------------
    # 5. KEEP
    # -----------------------------------------------------

    if (
        persistence < 0.20
        and context < 0.20
        and lagged < 0.20
        and interaction < 0.20
    ):
        return "KEEP"

    # -----------------------------------------------------
    # 6. INSUFFICIENT / AMBIGUOUS
    # -----------------------------------------------------

    return "ABSTAIN"


df["ETMR_Decision"] = df.apply(
    etmr_counterfactual,
    axis=1
)

df["Correct"] = (
    df["ETMR_Decision"] == df["Expected"]
)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("FULL ETMR RESULTS")
print("=" * 80)

print(
    df[
        [
            "World",
            "Expected",
            "ETMR_Decision",
            "Correct"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# ACCURACY
# ---------------------------------------------------------

accuracy = df["Correct"].mean()

print("\nCorrect:", df["Correct"].sum())
print("Total  :", len(df))
print("Accuracy:", f"{accuracy:.3f}")

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(
    pd.crosstab(
        df["Expected"],
        df["ETMR_Decision"]
    )
)

# ---------------------------------------------------------
# REVISION SAFETY
# ---------------------------------------------------------

false_revision = (
    (df["ETMR_Decision"] == "REVISE") &
    (df["Expected"] != "REVISE")
).sum()

true_revision = (
    (df["ETMR_Decision"] == "REVISE") &
    (df["Expected"] == "REVISE")
).sum()

actual_revision = (
    df["Expected"] == "REVISE"
).sum()

missed_revision = (
    (df["Expected"] == "REVISE") &
    (df["ETMR_Decision"] != "REVISE")
).sum()

non_revision = (
    df["Expected"] != "REVISE"
).sum()

print("\n" + "=" * 80)
print("REVISION SAFETY")
print("=" * 80)

print("True revisions :", true_revision)
print("False revisions:", false_revision)
print(
    "False revision rate:",
    f"{false_revision / non_revision:.3f}"
)

print("Missed revisions:", missed_revision)
print(
    "Missed revision rate:",
    f"{missed_revision / actual_revision:.3f}"
)

# ---------------------------------------------------------
# PER-CLASS ACCURACY
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("PER-CLASS PERFORMANCE")
print("=" * 80)

for label in [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]:

    actual = (df["Expected"] == label).sum()
    correct = (
        (df["Expected"] == label) &
        (df["ETMR_Decision"] == label)
    ).sum()

    print(
        f"{label:12s}: {correct}/{actual}"
    )

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv(
    "step49_full_counterfactual_etmr.csv",
    index=False
)

print("\nSaved:")
print("step49_full_counterfactual_etmr.csv")

print("\n" + "=" * 80)
print("STEP 49 COMPLETE")
print("=" * 80)
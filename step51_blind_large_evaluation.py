import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 51: BLIND EVALUATION ON LARGE INDEPENDENT BENCHMARK")
print("=" * 80)

# ============================================================
# 1. Load independent benchmark
# ============================================================

df = pd.read_csv("step50_large_independent_benchmark.csv")

print("\nLoaded cases:", len(df))


# ============================================================
# 2. FROZEN STEP 49 POLICY
#    DO NOT CHANGE THESE THRESHOLDS
# ============================================================

QUALITY_THRESHOLD = 0.80
PARAMETER_THRESHOLD = 0.70
STRUCTURAL_THRESHOLD = 0.30


# ============================================================
# 3. Counterfactual evidence calculations
# ============================================================

# How well can persistence explain the discrepancy
# without requiring a structural mechanism?
df["Parameter_Gain"] = (
    df["Temporal"]
    * (1 - df["Lagged_Rainfall"])
    * (1 - df["Interaction"])
)

# Initial structural evidence after accounting for
# the parameter explanation
df["Structural_Gain"] = (
    df["Context"]
    * (1 - df["Parameter_Gain"])
)

# Strong independent structural mechanisms
df["Structural_Gain"] = np.maximum(
    df["Structural_Gain"],
    df["Lagged_Rainfall"]
)

df["Structural_Gain"] = np.maximum(
    df["Structural_Gain"],
    df["Interaction"]
)


# ============================================================
# 4. FROZEN DECISION RULE
# ============================================================

def etmr_decision(row):

    q = row["Q_evidence"]
    parameter_gain = row["Parameter_Gain"]
    structural_gain = row["Structural_Gain"]

    # Insufficient evidence
    if q < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # Strong parameter explanation and weak
    # independent structural evidence
    if (
        parameter_gain >= PARAMETER_THRESHOLD
        and structural_gain < STRUCTURAL_THRESHOLD
    ):
        return "RECALIBRATE"

    # Strong structural evidence
    if structural_gain >= STRUCTURAL_THRESHOLD:
        return "REVISE"

    # Weak evidence for all alternatives
    if (
        parameter_gain < 0.30
        and structural_gain < 0.30
    ):
        return "KEEP"

    return "ABSTAIN"


df["Predicted"] = df.apply(etmr_decision, axis=1)


# ============================================================
# 5. Overall accuracy
# ============================================================

accuracy = (
    df["Predicted"] == df["Expected"]
).mean()

print("\n" + "-" * 80)
print("OVERALL RESULT")
print("-" * 80)

print(f"Accuracy: {accuracy:.4f}")
print(f"Correct: {(df['Predicted'] == df['Expected']).sum()} / {len(df)}")


# ============================================================
# 6. Confusion matrix
# ============================================================

print("\nConfusion Matrix:")
confusion = pd.crosstab(
    df["Expected"],
    df["Predicted"],
    rownames=["Expected"],
    colnames=["Predicted"],
    dropna=False
)

print(confusion)


# ============================================================
# 7. Per-class accuracy
# ============================================================

print("\nPer-class accuracy:")

for cls in ["KEEP", "RECALIBRATE", "REVISE", "ABSTAIN"]:

    subset = df[df["Expected"] == cls]

    class_accuracy = (
        subset["Predicted"] == subset["Expected"]
    ).mean()

    print(
        f"{cls:15s}: "
        f"{class_accuracy:.4f} "
        f"({(subset['Predicted'] == subset['Expected']).sum()}/{len(subset)})"
    )


# ============================================================
# 8. Error analysis
# ============================================================

errors = df[df["Predicted"] != df["Expected"]].copy()

print("\n" + "-" * 80)
print("ERROR ANALYSIS")
print("-" * 80)

print("Number of errors:", len(errors))

if len(errors) > 0:

    print("\nErrors:")
    print(
        errors[
            [
                "World",
                "Expected",
                "Predicted",
                "Q_evidence",
                "Temporal",
                "Context",
                "Lagged_Rainfall",
                "Interaction",
                "Parameter_Gain",
                "Structural_Gain"
            ]
        ].to_string(index=False)
    )

else:
    print("\nNO ERRORS.")


# ============================================================
# 9. False revision / missed revision
# ============================================================

false_revision = (
    (
        (df["Predicted"] == "REVISE")
        & (df["Expected"] != "REVISE")
    ).sum()
)

missed_revision = (
    (
        (df["Expected"] == "REVISE")
        & (df["Predicted"] != "REVISE")
    ).sum()
)

false_recalibration = (
    (
        (df["Predicted"] == "RECALIBRATE")
        & (df["Expected"] != "RECALIBRATE")
    ).sum()
)

missed_recalibration = (
    (
        (df["Expected"] == "RECALIBRATE")
        & (df["Predicted"] != "RECALIBRATE")
    ).sum()
)

print("\n" + "-" * 80)
print("SCIENTIFIC DECISION ERRORS")
print("-" * 80)

print(f"False revisions:        {false_revision}")
print(f"Missed revisions:       {missed_revision}")
print(f"False recalibrations:   {false_recalibration}")
print(f"Missed recalibrations:  {missed_recalibration}")


# ============================================================
# 10. Save complete evaluation
# ============================================================

df.to_csv(
    "step51_blind_large_evaluation.csv",
    index=False
)

print("\nSaved:")
print("step51_blind_large_evaluation.csv")

print("\n" + "=" * 80)
print("STEP 51 COMPLETE")
print("=" * 80)
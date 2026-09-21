import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 53: FROZEN POLICY ON AMBIGUOUS STRESS TEST")
print("=" * 80)

# ============================================================
# 1. Load stress-test benchmark
# ============================================================

df = pd.read_csv("step52_ambiguous_stress_test.csv")

print("\nLoaded cases:", len(df))


# ============================================================
# 2. FROZEN STEP 49 POLICY
# ============================================================

QUALITY_THRESHOLD = 0.80
PARAMETER_THRESHOLD = 0.70
STRUCTURAL_THRESHOLD = 0.30


# ============================================================
# 3. Counterfactual parameter explanation
# ============================================================

df["Parameter_Gain"] = (
    df["Temporal"]
    * (1 - df["Lagged_Rainfall"])
    * (1 - df["Interaction"])
)


# ============================================================
# 4. Structural explanation
# ============================================================

df["Structural_Gain"] = (
    df["Context"]
    * (1 - df["Parameter_Gain"])
)

df["Structural_Gain"] = np.maximum(
    df["Structural_Gain"],
    df["Lagged_Rainfall"]
)

df["Structural_Gain"] = np.maximum(
    df["Structural_Gain"],
    df["Interaction"]
)


# ============================================================
# 5. FROZEN ETMR DECISION
# ============================================================

def etmr_decision(row):

    q = row["Q_evidence"]
    parameter_gain = row["Parameter_Gain"]
    structural_gain = row["Structural_Gain"]

    # Insufficient evidence
    if q < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # Parameter explanation dominates
    if (
        parameter_gain >= PARAMETER_THRESHOLD
        and structural_gain < STRUCTURAL_THRESHOLD
    ):
        return "RECALIBRATE"

    # Structural explanation dominates
    if structural_gain >= STRUCTURAL_THRESHOLD:
        return "REVISE"

    # Weak evidence
    if (
        parameter_gain < 0.30
        and structural_gain < 0.30
    ):
        return "KEEP"

    return "ABSTAIN"


df["Predicted"] = df.apply(etmr_decision, axis=1)


# ============================================================
# 6. Overall performance
# ============================================================

accuracy = (
    df["Predicted"] == df["Expected"]
).mean()

print("\n" + "-" * 80)
print("OVERALL RESULT")
print("-" * 80)

print(f"Accuracy: {accuracy:.4f}")
print(
    f"Correct: "
    f"{(df['Predicted'] == df['Expected']).sum()} / {len(df)}"
)


# ============================================================
# 7. Confusion matrix
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
# 8. Per-group analysis
# ============================================================

print("\n" + "-" * 80)
print("PER-GROUP ANALYSIS")
print("-" * 80)

groups = {
    "BOUNDARY_PARAMETER": df[
        df["World"].str.startswith("BOUNDARY_PARAMETER")
    ],
    "BOUNDARY_STRUCTURE": df[
        df["World"].str.startswith("BOUNDARY_STRUCTURE")
    ],
    "BOUNDARY_QUALITY": df[
        df["World"].str.startswith("BOUNDARY_QUALITY")
    ],
    "BOUNDARY_LAGGED": df[
        df["World"].str.startswith("BOUNDARY_LAGGED")
    ],
    "CONFLICTING": df[
        df["World"].str.startswith("CONFLICTING")
    ]
}

for name, subset in groups.items():

    correct = (
        subset["Predicted"] == subset["Expected"]
    ).sum()

    acc = correct / len(subset)

    print(
        f"{name:25s}: "
        f"{acc:.4f} "
        f"({correct}/{len(subset)})"
    )

    print(
        "  Predictions:",
        subset["Predicted"].value_counts().to_dict()
    )


# ============================================================
# 9. Show every error
# ============================================================

errors = df[
    df["Predicted"] != df["Expected"]
].copy()

print("\n" + "-" * 80)
print("ERROR ANALYSIS")
print("-" * 80)

print("Total errors:", len(errors))

if len(errors) > 0:

    print("\nMisclassified cases:")

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
# 10. False revision analysis
# ============================================================

false_revision = (
    (
        (df["Predicted"] == "REVISE")
        & (df["Expected"] != "REVISE")
    ).sum()
)

print("\n" + "-" * 80)
print("REVISION SAFETY")
print("-" * 80)

print("False revisions:", false_revision)

if false_revision == 0:
    print(
        "ETMR made no unjustified structural revisions "
        "on the ambiguous benchmark."
    )
else:
    print(
        "WARNING: ETMR produced unjustified structural revisions."
    )


# ============================================================
# 11. Abstention analysis
# ============================================================

correct_abstentions = (
    (
        (df["Predicted"] == "ABSTAIN")
        & (df["Expected"] == "ABSTAIN")
    ).sum()
)

print("\nCorrect abstentions:", correct_abstentions)

print(
    "Abstention rate:",
    (df["Predicted"] == "ABSTAIN").mean()
)


# ============================================================
# 12. Save results
# ============================================================

df.to_csv(
    "step53_ambiguous_evaluation.csv",
    index=False
)

print("\nSaved:")
print("step53_ambiguous_evaluation.csv")

print("\n" + "=" * 80)
print("STEP 53 COMPLETE")
print("=" * 80)
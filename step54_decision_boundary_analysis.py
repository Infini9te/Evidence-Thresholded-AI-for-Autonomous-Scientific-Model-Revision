import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 54: DECISION BOUNDARY ANALYSIS")
print("=" * 80)

# ============================================================
# 1. Load Step 53 results
# ============================================================

df = pd.read_csv("step53_ambiguous_evaluation.csv")

print("\nLoaded cases:", len(df))


# ============================================================
# 2. Analyze each expected class
# ============================================================

classes = [
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]

for cls in classes:

    subset = df[df["Expected"] == cls]

    print("\n" + "-" * 80)
    print(cls)
    print("-" * 80)

    if len(subset) == 0:
        continue

    columns = [
        "Parameter_Gain",
        "Structural_Gain",
        "Temporal",
        "Context",
        "Lagged_Rainfall",
        "Interaction",
        "Q_evidence"
    ]

    print(
        subset[columns].describe().loc[
            ["min", "25%", "50%", "75%", "max"]
        ].to_string()
    )


# ============================================================
# 3. Parameter decision boundary
# ============================================================

parameter_cases = df[
    df["Expected"] == "RECALIBRATE"
].copy()

print("\n" + "-" * 80)
print("PARAMETER CASES — CURRENT POLICY")
print("-" * 80)

print(
    parameter_cases[
        [
            "World",
            "Temporal",
            "Context",
            "Lagged_Rainfall",
            "Interaction",
            "Parameter_Gain",
            "Structural_Gain",
            "Predicted"
        ]
    ].to_string(index=False)
)


# ============================================================
# 4. What would happen at different parameter thresholds?
#    Diagnostic ONLY — no policy change.
# ============================================================

print("\n" + "-" * 80)
print("PARAMETER THRESHOLD SENSITIVITY")
print("-" * 80)

thresholds = np.arange(0.30, 0.76, 0.05)

for threshold in thresholds:

    correct = 0

    for _, row in parameter_cases.iterrows():

        parameter_gain = row["Parameter_Gain"]
        structural_gain = row["Structural_Gain"]

        if (
            parameter_gain >= threshold
            and structural_gain < 0.30
        ):
            prediction = "RECALIBRATE"
        elif structural_gain >= 0.30:
            prediction = "REVISE"
        else:
            prediction = "ABSTAIN"

        if prediction == "RECALIBRATE":
            correct += 1

    print(
        f"Threshold {threshold:.2f}: "
        f"{correct}/{len(parameter_cases)} "
        f"parameter cases classified as RECALIBRATE"
    )


# ============================================================
# 5. Find the strongest ABSTAIN cases
# ============================================================

abstain_cases = df[
    df["Expected"] == "RECALIBRATE"
].copy()

abstain_cases["Distance_to_parameter_threshold"] = (
    0.70 - abstain_cases["Parameter_Gain"]
)

print("\n" + "-" * 80)
print("PARAMETER CASES CLOSEST TO RECALIBRATE")
print("-" * 80)

print(
    abstain_cases[
        [
            "World",
            "Parameter_Gain",
            "Structural_Gain",
            "Distance_to_parameter_threshold"
        ]
    ]
    .sort_values("Distance_to_parameter_threshold")
    .to_string(index=False)
)


# ============================================================
# 6. Structural safety analysis
# ============================================================

false_revision = (
    (
        (df["Predicted"] == "REVISE")
        & (df["Expected"] != "REVISE")
    ).sum()
)

print("\n" + "-" * 80)
print("STRUCTURAL SAFETY")
print("-" * 80)

print("False revisions:", false_revision)

if false_revision == 0:
    print(
        "No false structural revisions occurred."
    )
else:
    print(
        "False structural revisions detected."
    )


# ============================================================
# 7. Save boundary diagnostics
# ============================================================

df.to_csv(
    "step54_decision_boundary_analysis.csv",
    index=False
)

print("\nSaved:")
print("step54_decision_boundary_analysis.csv")

print("\n" + "=" * 80)
print("STEP 54 COMPLETE")
print("=" * 80)
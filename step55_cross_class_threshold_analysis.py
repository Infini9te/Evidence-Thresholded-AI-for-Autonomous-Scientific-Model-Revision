import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 55: CROSS-CLASS PARAMETER THRESHOLD SAFETY ANALYSIS")
print("=" * 80)

# ============================================================
# 1. Load both independent evaluation datasets
# ============================================================

large = pd.read_csv("step51_blind_large_evaluation.csv")
ambiguous = pd.read_csv("step53_ambiguous_evaluation.csv")

df = pd.concat(
    [large, ambiguous],
    ignore_index=True
)

print("\nLarge benchmark cases:", len(large))
print("Ambiguous benchmark cases:", len(ambiguous))
print("Combined cases:", len(df))


# ============================================================
# 2. Test candidate parameter thresholds
# ============================================================

thresholds = np.arange(0.30, 0.76, 0.05)

results = []


for threshold in thresholds:

    predictions = []

    for _, row in df.iterrows():

        q = row["Q_evidence"]
        parameter_gain = row["Parameter_Gain"]
        structural_gain = row["Structural_Gain"]

        # --------------------------------------------
        # Evidence quality
        # --------------------------------------------

        if q < 0.80:
            prediction = "ABSTAIN"

        # --------------------------------------------
        # Parameter explanation
        # --------------------------------------------

        elif (
            parameter_gain >= threshold
            and structural_gain < 0.30
        ):
            prediction = "RECALIBRATE"

        # --------------------------------------------
        # Structural explanation
        # --------------------------------------------

        elif structural_gain >= 0.30:
            prediction = "REVISE"

        # --------------------------------------------
        # Weak evidence
        # --------------------------------------------

        elif (
            parameter_gain < 0.30
            and structural_gain < 0.30
        ):
            prediction = "KEEP"

        else:
            prediction = "ABSTAIN"

        predictions.append(prediction)

    df_temp = df.copy()
    df_temp["Threshold_Predicted"] = predictions

    # ========================================================
    # Metrics
    # ========================================================

    accuracy = (
        df_temp["Threshold_Predicted"]
        == df_temp["Expected"]
    ).mean()

    false_revision = (
        (
            (df_temp["Threshold_Predicted"] == "REVISE")
            & (df_temp["Expected"] != "REVISE")
        ).sum()
    )

    missed_revision = (
        (
            (df_temp["Expected"] == "REVISE")
            & (df_temp["Threshold_Predicted"] != "REVISE")
        ).sum()
    )

    false_recalibration = (
        (
            (df_temp["Threshold_Predicted"] == "RECALIBRATE")
            & (df_temp["Expected"] != "RECALIBRATE")
        ).sum()
    )

    missed_recalibration = (
        (
            (df_temp["Expected"] == "RECALIBRATE")
            & (df_temp["Threshold_Predicted"] != "RECALIBRATE")
        ).sum()
    )

    correct_recalibration = (
        (
            (df_temp["Expected"] == "RECALIBRATE")
            & (df_temp["Threshold_Predicted"] == "RECALIBRATE")
        ).sum()
    )

    correct_abstain = (
        (
            (df_temp["Expected"] == "ABSTAIN")
            & (df_temp["Threshold_Predicted"] == "ABSTAIN")
        ).sum()
    )

    results.append({
        "Parameter_Threshold": threshold,
        "Accuracy": accuracy,
        "Correct_RECALIBRATE": correct_recalibration,
        "Missed_RECALIBRATE": missed_recalibration,
        "False_RECALIBRATE": false_recalibration,
        "Correct_ABSTAIN": correct_abstain,
        "False_REVISE": false_revision,
        "Missed_REVISE": missed_revision
    })


# ============================================================
# 3. Display results
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "-" * 80)
print("CROSS-CLASS THRESHOLD RESULTS")
print("-" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Parameter_Threshold": "{:.2f}".format,
            "Accuracy": "{:.4f}".format
        }
    )
)


# ============================================================
# 4. Safety-first candidates
# ============================================================

print("\n" + "-" * 80)
print("SAFETY-FIRST ANALYSIS")
print("-" * 80)

safe = results_df[
    (results_df["False_REVISE"] == 0)
    & (results_df["Missed_REVISE"] == 0)
]

if len(safe) > 0:

    print(
        "Thresholds with ZERO false revisions "
        "and ZERO missed revisions:"
    )

    print(
        safe.to_string(index=False)
    )

else:

    print(
        "No threshold satisfies zero false "
        "and zero missed revisions."
    )


# ============================================================
# 5. Best recalibration recovery without false revisions
# ============================================================

candidate = results_df[
    results_df["False_REVISE"] == 0
].sort_values(
    [
        "Correct_RECALIBRATE",
        "Accuracy"
    ],
    ascending=False
)

print("\n" + "-" * 80)
print("BEST RECALIBRATION RECOVERY WITH ZERO FALSE REVISIONS")
print("-" * 80)

print(
    candidate.head(5).to_string(index=False)
)


# ============================================================
# 6. Save
# ============================================================

results_df.to_csv(
    "step55_cross_class_threshold_analysis.csv",
    index=False
)

print("\nSaved:")
print("step55_cross_class_threshold_analysis.csv")

print("\n" + "=" * 80)
print("STEP 55 COMPLETE")
print("=" * 80)
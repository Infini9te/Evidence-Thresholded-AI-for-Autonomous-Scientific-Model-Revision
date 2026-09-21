import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 45: EVIDENCE SUFFICIENCY CALIBRATION")
print("=" * 80)

# ---------------------------------------------------------
# LOAD STANDARDIZED BENCHMARK
# ---------------------------------------------------------

df = pd.read_csv("step42_standardized_evidence.csv")

print("\nTotal benchmark cases:", len(df))

# ---------------------------------------------------------
# DECISION FUNCTION
# ---------------------------------------------------------

def etmr_decision(row, quality_threshold):

    q = row["Q_evidence"]
    persistence = row["Temporal"]
    context = row["Context"]
    lagged = row["Lagged_Rainfall"]
    interaction = row["Interaction"]

    # Evidence sufficiency gate
    if q < quality_threshold:
        return "ABSTAIN"

    # Strong persistent discrepancy without independent
    # structural evidence -> parameter recalibration
    if (
        persistence >= 0.70
        and lagged < 0.70
        and interaction < 0.60
        and context < 0.70
    ):
        return "RECALIBRATE"

    # Structural evidence
    if context >= 0.70 and persistence >= 0.20:
        return "REVISE"

    if lagged >= 0.70:
        return "REVISE"

    if interaction >= 0.60:
        return "REVISE"

    # Weak discrepancy -> KEEP
    if (
        persistence < 0.20
        and context < 0.20
        and lagged < 0.20
        and interaction < 0.20
    ):
        return "KEEP"

    return "ABSTAIN"


# ---------------------------------------------------------
# THRESHOLD SWEEP
# ---------------------------------------------------------

thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)

results = []

for threshold in thresholds:

    predictions = df.apply(
        lambda row: etmr_decision(row, threshold),
        axis=1
    )

    expected = df["Expected"]

    accuracy = (predictions == expected).mean()

    # False revision:
    # predicted REVISE when expected != REVISE
    false_revision = (
        (predictions == "REVISE") &
        (expected != "REVISE")
    ).sum()

    non_revision_cases = (expected != "REVISE").sum()

    false_revision_rate = (
        false_revision / non_revision_cases
        if non_revision_cases > 0 else 0
    )

    # Missed revision:
    # expected REVISE but prediction != REVISE
    missed_revision = (
        (expected == "REVISE") &
        (predictions != "REVISE")
    ).sum()

    revision_cases = (expected == "REVISE").sum()

    missed_revision_rate = (
        missed_revision / revision_cases
        if revision_cases > 0 else 0
    )

    results.append({
        "Quality_Threshold": threshold,
        "Accuracy": accuracy,
        "False_Revision_Rate": false_revision_rate,
        "Missed_Revision_Rate": missed_revision_rate,
        "KEEP_Correct": (
            (predictions == "KEEP") &
            (expected == "KEEP")
        ).sum(),
        "RECALIBRATE_Correct": (
            (predictions == "RECALIBRATE") &
            (expected == "RECALIBRATE")
        ).sum(),
        "REVISE_Correct": (
            (predictions == "REVISE") &
            (expected == "REVISE")
        ).sum(),
        "ABSTAIN_Correct": (
            (predictions == "ABSTAIN") &
            (expected == "ABSTAIN")
        ).sum()
    })


results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("THRESHOLD CALIBRATION RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.3f}".format,
            "False_Revision_Rate": "{:.3f}".format,
            "Missed_Revision_Rate": "{:.3f}".format
        }
    )
)

# ---------------------------------------------------------
# FIND SAFEST THRESHOLD
# ---------------------------------------------------------

safe = results_df[
    (results_df["False_Revision_Rate"] == 0.0) &
    (results_df["Missed_Revision_Rate"] == 0.0)
]

print("\n" + "=" * 80)
print("ZERO-FALSE-REVISION / ZERO-MISSED-REVISION REGION")
print("=" * 80)

if len(safe) > 0:
    print(safe.to_string(index=False))
else:
    print("No threshold achieves both zero false revisions and zero missed revisions.")

# ---------------------------------------------------------
# BEST ACCURACY
# ---------------------------------------------------------

best_accuracy = results_df[
    results_df["Accuracy"] == results_df["Accuracy"].max()
]

print("\n" + "=" * 80)
print("BEST ACCURACY")
print("=" * 80)

print(best_accuracy.to_string(index=False))

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

results_df.to_csv(
    "step45_evidence_threshold_calibration.csv",
    index=False
)

print("\nSaved:")
print("step45_evidence_threshold_calibration.csv")

print("\n" + "=" * 80)
print("STEP 45 COMPLETE")
print("=" * 80)
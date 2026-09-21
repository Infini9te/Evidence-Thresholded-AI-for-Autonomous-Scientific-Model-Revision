import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 83: FORMAL SAFETY AND ERROR-COST ANALYSIS")
print("=" * 80)

# ============================================================
# LOAD FINAL FROZEN EVALUATION
# ============================================================

file_path = "step82_final_frozen_evaluation_results.csv"

df = pd.read_csv(file_path)

print("\nLoaded:")
print(file_path)

print(f"Number of cases = {len(df)}")


# ============================================================
# BASIC ERROR COUNTS
# ============================================================

df["correct"] = (
    df["truth"] == df["predicted"]
)

false_revision = (
    (df["predicted"] == "REVISE") &
    (df["truth"] != "REVISE")
)

missed_revision = (
    (df["truth"] == "REVISE") &
    (df["predicted"] != "REVISE")
)

false_recalibration = (
    (df["predicted"] == "RECALIBRATE") &
    (df["truth"] != "RECALIBRATE")
)

unnecessary_change = (
    (df["truth"] == "KEEP") &
    (df["predicted"] != "KEEP")
)

incorrect_abstention = (
    (df["predicted"] == "ABSTAIN") &
    (df["truth"] != "ABSTAIN")
)

correct_abstention = (
    (df["predicted"] == "ABSTAIN") &
    (df["truth"] == "ABSTAIN")
)


# ============================================================
# RATES
# ============================================================

accuracy = df["correct"].mean()

false_revision_rate = (
    false_revision.sum()
    / (df["truth"] != "REVISE").sum()
)

missed_revision_rate = (
    missed_revision.sum()
    / (df["truth"] == "REVISE").sum()
)

false_recalibration_rate = (
    false_recalibration.sum()
    / (df["truth"] != "RECALIBRATE").sum()
)

unnecessary_change_rate = (
    unnecessary_change.sum()
    / (df["truth"] == "KEEP").sum()
)

abstention_precision = (
    correct_abstention.sum()
    / (df["predicted"] == "ABSTAIN").sum()
)

abstention_recall = (
    correct_abstention.sum()
    / (df["truth"] == "ABSTAIN").sum()
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("CORE SAFETY METRICS")
print("=" * 80)

print(f"Overall accuracy          = {accuracy:.4f}")
print(f"False revision rate       = {false_revision_rate:.4f}")
print(f"Missed revision rate      = {missed_revision_rate:.4f}")
print(f"False recalibration rate  = {false_recalibration_rate:.4f}")
print(f"Unnecessary change rate   = {unnecessary_change_rate:.4f}")
print(f"ABSTAIN precision         = {abstention_precision:.4f}")
print(f"ABSTAIN recall            = {abstention_recall:.4f}")


# ============================================================
# ERROR COST
#
# These are NOT learned weights.
# They are an illustrative scientific-risk analysis.
#
# False REVISE is assigned highest cost because unnecessary
# structural model change is the most consequential action.
# ============================================================

costs = {
    "false_revise": 5,
    "missed_revise": 3,
    "false_recalibrate": 2,
    "unnecessary_change": 2,
    "incorrect_abstain": 1
}

total_cost = (
    false_revision.sum() * costs["false_revise"] +
    missed_revision.sum() * costs["missed_revise"] +
    false_recalibration.sum() * costs["false_recalibrate"] +
    unnecessary_change.sum() * costs["unnecessary_change"] +
    incorrect_abstention.sum() * costs["incorrect_abstain"]
)

print("\n" + "=" * 80)
print("ERROR-COST ANALYSIS")
print("=" * 80)

print("Illustrative scientific-risk weights:")
print(f"False REVISE       = {costs['false_revise']}")
print(f"Missed REVISE      = {costs['missed_revise']}")
print(f"False RECALIBRATE  = {costs['false_recalibrate']}")
print(f"Unnecessary change = {costs['unnecessary_change']}")
print(f"Incorrect ABSTAIN  = {costs['incorrect_abstain']}")

print(f"\nTotal error cost = {total_cost}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

confusion = pd.crosstab(
    df["truth"],
    df["predicted"]
)

print(confusion)


# ============================================================
# ERROR LIST
# ============================================================

errors = df[~df["correct"]]

print("\n" + "=" * 80)
print("ERROR CASES")
print("=" * 80)

if len(errors) == 0:
    print("No errors in the final frozen benchmark.")
else:
    print(
        errors[
            [
                "case",
                "truth",
                "predicted",
                "alpha_case",
                "structural_relative_gain",
                "quality"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SCIENTIFIC SAFETY STATEMENT
# ============================================================

print("\n" + "=" * 80)
print("SCIENTIFIC SAFETY CHECK")
print("=" * 80)

if false_revision.sum() == 0:
    print(
        "PASS: ETMR produced no unjustified structural revisions "
        "in the final frozen benchmark."
    )
else:
    print(
        "WARNING: false structural revisions occurred."
    )

if missed_revision.sum() == 0:
    print(
        "PASS: ETMR missed no planted structural revisions "
        "in the final frozen benchmark."
    )
else:
    print(
        "WARNING: structural revisions were missed."
    )

if unnecessary_change.sum() == 0:
    print(
        "PASS: no KEEP case was unnecessarily changed."
    )
else:
    print(
        "WARNING: unnecessary model changes occurred."
    )


# ============================================================
# SAVE
# ============================================================

output_file = "step83_safety_analysis_results.csv"

summary = pd.DataFrame([{
    "accuracy": accuracy,
    "false_revision_rate": false_revision_rate,
    "missed_revision_rate": missed_revision_rate,
    "false_recalibration_rate": false_recalibration_rate,
    "unnecessary_change_rate": unnecessary_change_rate,
    "abstain_precision": abstention_precision,
    "abstain_recall": abstention_recall,
    "total_error_cost": total_cost
}])

summary.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 80)
print("STEP 83 COMPLETE")
print("=" * 80)
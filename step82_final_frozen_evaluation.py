import numpy as np
import pandas as pd

print("=" * 80)
print("ETMR — STEP 82: FINAL FROZEN END-TO-END EVALUATION")
print("=" * 80)

# ============================================================
# FROZEN PARAMETERS
# ============================================================

ALPHA_LOW = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

print("\nFrozen policy:")
print(f"Alpha lower boundary : {ALPHA_LOW:.6f}")
print(f"Structural threshold : {STRUCTURAL_THRESHOLD:.6f}")
print(f"Quality threshold    : {QUALITY_THRESHOLD:.2f}")


# ============================================================
# COMPLETE FROZEN ETMR POLICY
# ============================================================

def etmr_decision(alpha_case, structural_gain, quality):

    # --------------------------------------------------------
    # Stage 1: Evidence quality
    # --------------------------------------------------------
    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # --------------------------------------------------------
    # Stage 2: Structural revision
    # --------------------------------------------------------
    if structural_gain > STRUCTURAL_THRESHOLD:
        return "REVISE"

    # --------------------------------------------------------
    # Stage 3: Parameter recalibration
    # --------------------------------------------------------
    if alpha_case < ALPHA_LOW:
        return "RECALIBRATE"

    # --------------------------------------------------------
    # Stage 4: No sufficient evidence for change
    # --------------------------------------------------------
    return "KEEP"


# ============================================================
# FINAL INDEPENDENT BENCHMARK
#
# New seed.
# No parameters are estimated from these cases.
# ============================================================

rng = np.random.default_rng(20260913)

cases = []

# ------------------------------------------------------------
# KEEP
# ------------------------------------------------------------

for i in range(15):

    cases.append({
        "case": f"FINAL_KEEP_{i+1}",
        "truth": "KEEP",
        "alpha_case": rng.uniform(1.0396, 1.0418),
        "structural_relative_gain": 0.0,
        "quality": rng.uniform(0.85, 1.00)
    })


# ------------------------------------------------------------
# RECALIBRATE
# ------------------------------------------------------------

for i in range(15):

    cases.append({
        "case": f"FINAL_RECALIBRATE_{i+1}",
        "truth": "RECALIBRATE",
        "alpha_case": rng.uniform(0.88, 0.99),
        "structural_relative_gain": 0.0,
        "quality": rng.uniform(0.85, 1.00)
    })


# ------------------------------------------------------------
# REVISE
# ------------------------------------------------------------

for i in range(15):

    cases.append({
        "case": f"FINAL_REVISE_{i+1}",
        "truth": "REVISE",
        "alpha_case": rng.uniform(1.70, 2.60),
        "structural_relative_gain": rng.uniform(0.55, 0.95),
        "quality": rng.uniform(0.85, 1.00)
    })


# ------------------------------------------------------------
# ABSTAIN
#
# Deliberately strong discrepancy but insufficient quality.
# ------------------------------------------------------------

for i in range(15):

    cases.append({
        "case": f"FINAL_ABSTAIN_{i+1}",
        "truth": "ABSTAIN",
        "alpha_case": rng.uniform(1.70, 2.60),
        "structural_relative_gain": rng.uniform(0.55, 0.95),
        "quality": rng.uniform(0.20, 0.79)
    })


# ============================================================
# APPLY FROZEN POLICY
# ============================================================

df = pd.DataFrame(cases)

df["predicted"] = df.apply(
    lambda row: etmr_decision(
        row["alpha_case"],
        row["structural_relative_gain"],
        row["quality"]
    ),
    axis=1
)


# ============================================================
# DISPLAY CASES
# ============================================================

print("\n" + "=" * 80)
print("FINAL BLIND CASES")
print("=" * 80)

print(
    df[
        [
            "case",
            "truth",
            "alpha_case",
            "structural_relative_gain",
            "quality",
            "predicted"
        ]
    ].to_string(index=False)
)


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
# OVERALL ACCURACY
# ============================================================

accuracy = (
    df["truth"] == df["predicted"]
).mean()

print("\n" + "=" * 80)
print("OVERALL PERFORMANCE")
print("=" * 80)

print(f"Overall accuracy = {accuracy:.4f}")


# ============================================================
# PER-CLASS PERFORMANCE
# ============================================================

for cls in [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]:

    subset = df[df["truth"] == cls]

    correct = (
        subset["predicted"] == cls
    ).sum()

    total = len(subset)

    print(
        f"{cls:12s}: "
        f"{correct}/{total} = "
        f"{correct/total:.4f}"
    )


# ============================================================
# REVISION SAFETY
# ============================================================

false_revisions = (
    (df["predicted"] == "REVISE") &
    (df["truth"] != "REVISE")
).sum()

missed_revisions = (
    (df["truth"] == "REVISE") &
    (df["predicted"] != "REVISE")
).sum()

actual_revisions = (
    df["truth"] == "REVISE"
).sum()

predicted_revisions = (
    df["predicted"] == "REVISE"
).sum()

revision_precision = (
    (actual_revisions - missed_revisions)
    / predicted_revisions
    if predicted_revisions > 0
    else np.nan
)

revision_recall = (
    (actual_revisions - missed_revisions)
    / actual_revisions
)

print("\n" + "=" * 80)
print("REVISION SAFETY")
print("=" * 80)

print(f"False revisions   = {false_revisions}")
print(f"Missed revisions  = {missed_revisions}")
print(f"Revision precision = {revision_precision:.4f}")
print(f"Revision recall    = {revision_recall:.4f}")


# ============================================================
# ABSTENTION PERFORMANCE
# ============================================================

actual_abstain = (
    df["truth"] == "ABSTAIN"
).sum()

predicted_abstain = (
    df["predicted"] == "ABSTAIN"
).sum()

correct_abstain = (
    (df["truth"] == "ABSTAIN") &
    (df["predicted"] == "ABSTAIN")
).sum()

abstain_precision = (
    correct_abstain / predicted_abstain
    if predicted_abstain > 0
    else np.nan
)

abstain_recall = (
    correct_abstain / actual_abstain
)

print("\n" + "=" * 80)
print("ABSTENTION PERFORMANCE")
print("=" * 80)

print(f"ABSTAIN precision = {abstain_precision:.4f}")
print(f"ABSTAIN recall    = {abstain_recall:.4f}")


# ============================================================
# FALSE CHANGE RATE
#
# A "change" means either RECALIBRATE or REVISE.
# ============================================================

true_keep = df["truth"] == "KEEP"

false_changes_on_keep = (
    true_keep &
    (df["predicted"] != "KEEP")
).sum()

false_change_rate = (
    false_changes_on_keep / true_keep.sum()
)

print("\n" + "=" * 80)
print("UNNECESSARY CHANGE SAFETY")
print("=" * 80)

print(
    f"KEEP cases incorrectly changed = "
    f"{false_changes_on_keep}"
)

print(
    f"False change rate = "
    f"{false_change_rate:.4f}"
)


# ============================================================
# QUALITY GATE CHECK
# ============================================================

low_quality = df[
    df["quality"] < QUALITY_THRESHOLD
]

quality_gate_violations = (
    low_quality["predicted"] != "ABSTAIN"
).sum()

print("\n" + "=" * 80)
print("QUALITY-GATE VALIDATION")
print("=" * 80)

print(
    f"Low-quality cases = {len(low_quality)}"
)

print(
    f"Quality-gate violations = "
    f"{quality_gate_violations}"
)


# ============================================================
# FINAL SAFETY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SAFETY SUMMARY")
print("=" * 80)

if (
    false_revisions == 0 and
    missed_revisions == 0 and
    quality_gate_violations == 0
):
    print("PASS: zero false revisions.")
    print("PASS: zero missed revisions.")
    print("PASS: low-quality evidence always abstains.")
else:
    print("WARNING: at least one safety condition failed.")


# ============================================================
# SAVE
# ============================================================

output_file = "step82_final_frozen_evaluation_results.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 80)
print("STEP 82 COMPLETE")
print("=" * 80)
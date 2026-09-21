import numpy as np
import pandas as pd

print("=" * 80)
print("ETMR — STEP 79: FROZEN ABSTAIN / EVIDENCE-QUALITY GATE")
print("=" * 80)

# ============================================================
# FROZEN PARAMETERS — DO NOT TUNE AFTER SEEING RESULTS
# ============================================================

ALPHA_LOW = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

print("\nFrozen rule:")
print(f"KEEP alpha lower boundary : {ALPHA_LOW:.6f}")
print(f"Structural threshold      : {STRUCTURAL_THRESHOLD:.6f}")
print(f"Quality threshold         : {QUALITY_THRESHOLD:.2f}")

# ============================================================
# DECISION FUNCTION
# ============================================================

def etmr_decision(alpha_case, structural_gain, quality):
    # Evidence quality has priority.
    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # Strong structural counterfactual advantage.
    if structural_gain > STRUCTURAL_THRESHOLD:
        return "REVISE"

    # Parameter deviation below the accepted KEEP interval.
    if alpha_case < ALPHA_LOW:
        return "RECALIBRATE"

    return "KEEP"


# ============================================================
# INDEPENDENT ABSTAIN STRESS CASES
# ============================================================

rng = np.random.default_rng(20260912)

cases = []

# ------------------------------------------------------------
# 1. TRUE KEEP
# ------------------------------------------------------------

for i in range(10):
    alpha = rng.uniform(1.0397, 1.0418)
    structural = 0.0
    quality = rng.uniform(0.85, 1.00)

    cases.append({
        "case": f"AB_KEEP_{i+1}",
        "truth": "KEEP",
        "alpha_case": alpha,
        "structural_relative_gain": structural,
        "quality": quality
    })


# ------------------------------------------------------------
# 2. TRUE RECALIBRATE
# ------------------------------------------------------------

for i in range(10):
    alpha = rng.uniform(0.88, 0.99)
    structural = 0.0
    quality = rng.uniform(0.85, 1.00)

    cases.append({
        "case": f"AB_RECALIBRATE_{i+1}",
        "truth": "RECALIBRATE",
        "alpha_case": alpha,
        "structural_relative_gain": structural,
        "quality": quality
    })


# ------------------------------------------------------------
# 3. TRUE REVISE — HIGH QUALITY
# ------------------------------------------------------------

for i in range(10):
    alpha = rng.uniform(1.70, 2.60)
    structural = rng.uniform(0.70, 0.95)
    quality = rng.uniform(0.85, 1.00)

    cases.append({
        "case": f"AB_REVISE_HIGHQ_{i+1}",
        "truth": "REVISE",
        "alpha_case": alpha,
        "structural_relative_gain": structural,
        "quality": quality
    })


# ------------------------------------------------------------
# 4. TRUE ABSTAIN — SAME STRONG DISCREPANCY,
#    BUT LOW EVIDENCE QUALITY
# ------------------------------------------------------------

for i in range(10):
    alpha = rng.uniform(1.70, 2.60)
    structural = rng.uniform(0.70, 0.95)
    quality = rng.uniform(0.20, 0.79)

    cases.append({
        "case": f"AB_ABSTAIN_{i+1}",
        "truth": "ABSTAIN",
        "alpha_case": alpha,
        "structural_relative_gain": structural,
        "quality": quality
    })


# ============================================================
# RUN FROZEN DECISION
# ============================================================

df = pd.DataFrame(cases)

df["predicted"] = df.apply(
    lambda r: etmr_decision(
        r["alpha_case"],
        r["structural_relative_gain"],
        r["quality"]
    ),
    axis=1
)

print("\n" + "=" * 80)
print("CASES")
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
# PERFORMANCE
# ============================================================

accuracy = (df["truth"] == df["predicted"]).mean()

print("\n" + "=" * 80)
print("PERFORMANCE")
print("=" * 80)

print(f"Overall accuracy = {accuracy:.4f}")

confusion = pd.crosstab(
    df["truth"],
    df["predicted"]
)

print("\nConfusion matrix:")
print(confusion)

for cls in ["KEEP", "RECALIBRATE", "REVISE", "ABSTAIN"]:
    subset = df[df["truth"] == cls]
    correct = (subset["predicted"] == cls).sum()
    total = len(subset)

    print(
        f"{cls:12s}: {correct}/{total} = "
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

print("\n" + "=" * 80)
print("REVISION SAFETY")
print("=" * 80)

print(f"False revisions = {false_revisions}")
print(f"Missed revisions = {missed_revisions}")


# ============================================================
# ABSTAIN SAFETY
# ============================================================

abstain_cases = df[df["truth"] == "ABSTAIN"]

abstain_correct = (
    abstain_cases["predicted"] == "ABSTAIN"
).sum()

abstain_precision_cases = df[
    df["predicted"] == "ABSTAIN"
]

if len(abstain_precision_cases) > 0:
    abstain_precision = (
        abstain_precision_cases["truth"] == "ABSTAIN"
    ).mean()
else:
    abstain_precision = np.nan

print("\n" + "=" * 80)
print("ABSTAIN SAFETY")
print("=" * 80)

print(
    f"ABSTAIN recall = "
    f"{abstain_correct}/{len(abstain_cases)} = "
    f"{abstain_correct/len(abstain_cases):.4f}"
)

print(
    f"ABSTAIN precision = "
    f"{abstain_precision:.4f}"
)


# ============================================================
# KEY SCIENTIFIC CHECK
# ============================================================

low_quality = df[df["quality"] < QUALITY_THRESHOLD]

violations = (
    low_quality["predicted"] != "ABSTAIN"
).sum()

print("\n" + "=" * 80)
print("QUALITY-GATE CHECK")
print("=" * 80)

print(
    f"Low-quality cases = {len(low_quality)}"
)

print(
    f"Low-quality cases not abstained = {violations}"
)

if violations == 0:
    print("PASS: low-quality evidence always triggers ABSTAIN.")
else:
    print("FAIL: quality gate did not enforce ABSTAIN.")


# ============================================================
# SAVE
# ============================================================

output_file = "step79_abstain_gate_results.csv"
df.to_csv(output_file, index=False)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 80)
print("STEP 79 COMPLETE")
print("=" * 80)
import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 70: BLIND NORMALIZED DECISION")
print("=" * 80)

INPUT = "step69_normalized_counterfactual_results.csv"

df = pd.read_csv(INPUT)

# ============================================================
# FROZEN DECISION POLICY
# ============================================================

QUALITY_THRESHOLD = 0.80

# Parameter evidence:
# positive improvement is required.
#
# We deliberately use a small threshold because the normalized
# gains in Step 69 are on the order of 0.01 rather than 0.30.
#
# More importantly, parameter correction must beat the
# structural explanation.

PARAMETER_THRESHOLD = 0.001

# Structural evidence threshold
STRUCTURAL_THRESHOLD = 0.30

# ============================================================
# DECISION
# ============================================================

decisions = []

for _, row in df.iterrows():

    quality = row["quality"]
    pg = row["parameter_relative_gain"]
    sg = row["structural_relative_gain"]

    # --------------------------------------------------------
    # 1. Measurement / provenance quality
    # --------------------------------------------------------

    if quality < QUALITY_THRESHOLD:
        decision = "ABSTAIN"

    # --------------------------------------------------------
    # 2. Structural counterfactual
    # --------------------------------------------------------

    elif (
        sg >= STRUCTURAL_THRESHOLD
        and sg > pg
    ):
        decision = "REVISE"

    # --------------------------------------------------------
    # 3. Parameter counterfactual
    # --------------------------------------------------------

    elif (
        pg >= PARAMETER_THRESHOLD
        and pg > sg
    ):
        decision = "RECALIBRATE"

    # --------------------------------------------------------
    # 4. No sufficient improvement
    # --------------------------------------------------------

    else:
        decision = "KEEP"

    decisions.append(decision)

df["decision"] = decisions

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("BLIND DECISIONS")
print("=" * 80)

print(
    df[
        [
            "case",
            "truth",
            "decision",
            "quality",
            "parameter_relative_gain",
            "structural_relative_gain"
        ]
    ].to_string(index=False)
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

cm = pd.crosstab(
    df["truth"],
    df["decision"],
    dropna=False
)

print(cm)

# ============================================================
# METRICS
# ============================================================

accuracy = np.mean(
    df["truth"] == df["decision"]
)

false_revision = np.mean(
    (df["decision"] == "REVISE") &
    (df["truth"] != "REVISE")
)

missed_revision = np.mean(
    (df["truth"] == "REVISE") &
    (df["decision"] != "REVISE")
)

abstain_precision = np.nan

if (df["decision"] == "ABSTAIN").sum() > 0:
    abstain_precision = np.mean(
        df.loc[
            df["decision"] == "ABSTAIN",
            "truth"
        ] == "ABSTAIN"
    )

print(f"\nOverall accuracy       = {accuracy:.4f}")
print(f"False revision rate   = {false_revision:.4f}")
print(f"Missed revision rate  = {missed_revision:.4f}")
print(f"ABSTAIN precision     = {abstain_precision:.4f}")

# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 80)
print("PER-CLASS ACCURACY")
print("=" * 80)

for cls in [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]:

    subset = df[df["truth"] == cls]

    acc = np.mean(
        subset["decision"] == subset["truth"]
    )

    print(
        f"{cls:15s}: "
        f"{acc:.4f} "
        f"({int(acc * len(subset))}/{len(subset)})"
    )

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "step70_blind_normalized_decisions.csv",
    index=False
)

print("\nSaved:")
print("step70_blind_normalized_decisions.csv")

print("\n" + "=" * 80)
print("STEP 70 COMPLETE")
print("=" * 80)
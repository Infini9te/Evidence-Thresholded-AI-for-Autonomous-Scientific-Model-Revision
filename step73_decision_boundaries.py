import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 73: COMPETING-EVIDENCE DECISION BOUNDARIES")
print("=" * 80)

df = pd.read_csv("step72_competing_evidence_results.csv")

# ---------------------------------------------------------
# 1. Inspect the two core evidence dimensions
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CORE EVIDENCE")
print("=" * 80)

cols = [
    "case",
    "truth",
    "parameter_relative_gain",
    "structural_relative_gain",
    "structural_advantage"
]

print(df[cols].to_string(index=False))

# ---------------------------------------------------------
# 2. Class ranges
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CLASS RANGES")
print("=" * 80)

for cls in ["KEEP", "RECALIBRATE", "REVISE"]:

    sub = df[df["truth"] == cls]

    print(f"\n{cls}")

    for col in [
        "parameter_relative_gain",
        "structural_relative_gain",
        "structural_advantage"
    ]:
        print(
            f"  {col}: "
            f"min={sub[col].min():.6f}, "
            f"mean={sub[col].mean():.6f}, "
            f"max={sub[col].max():.6f}"
        )

# ---------------------------------------------------------
# 3. Pairwise interval overlap
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("PAIRWISE SEPARATION")
print("=" * 80)

pairs = [
    ("RECALIBRATE", "KEEP"),
    ("KEEP", "REVISE"),
    ("RECALIBRATE", "REVISE")
]

for a, b in pairs:

    a_vals = df[df["truth"] == a]["structural_advantage"]
    b_vals = df[df["truth"] == b]["structural_advantage"]

    a_max = a_vals.max()
    b_min = b_vals.min()

    gap = b_min - a_max

    print(
        f"{a} vs {b}: "
        f"max({a})={a_max:.6f}, "
        f"min({b})={b_min:.6f}, "
        f"gap={gap:.6f}"
    )

# ---------------------------------------------------------
# 4. Natural midpoint boundaries
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("NATURAL MIDPOINT BOUNDARIES")
print("=" * 80)

# Boundary between RECALIBRATE and KEEP
rec_max = df[df["truth"] == "RECALIBRATE"]["structural_advantage"].max()
keep_min = df[df["truth"] == "KEEP"]["structural_advantage"].min()
keep_max = df[df["truth"] == "KEEP"]["structural_advantage"].max()
rev_min = df[df["truth"] == "REVISE"]["structural_advantage"].min()

boundary_rec_keep = (rec_max + keep_min) / 2
boundary_keep_rev = (keep_max + rev_min) / 2

print(f"RECALIBRATE / KEEP boundary = {boundary_rec_keep:.6f}")
print(f"KEEP / REVISE boundary       = {boundary_keep_rev:.6f}")

# ---------------------------------------------------------
# 5. Apply boundaries WITHOUT tuning
# ---------------------------------------------------------

def classify(x):

    if x <= boundary_rec_keep:
        return "RECALIBRATE"

    elif x <= boundary_keep_rev:
        return "KEEP"

    else:
        return "REVISE"


df["predicted"] = df["structural_advantage"].apply(classify)

print("\n" + "=" * 80)
print("DECISIONS FROM NATURAL BOUNDARIES")
print("=" * 80)

print(
    df[
        ["case", "truth", "structural_advantage", "predicted"]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# 6. Accuracy
# ---------------------------------------------------------

accuracy = (df["truth"] == df["predicted"]).mean()

print("\n" + "=" * 80)
print("PERFORMANCE")
print("=" * 80)

print(f"Accuracy = {accuracy:.4f}")

print("\nConfusion matrix:")
print(
    pd.crosstab(
        df["truth"],
        df["predicted"],
        rownames=["TRUE"],
        colnames=["PREDICTED"]
    )
)

# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------

df.to_csv(
    "step73_decision_boundaries_results.csv",
    index=False
)

print("\nSaved:")
print("step73_decision_boundaries_results.csv")

print("\n" + "=" * 80)
print("STEP 73 COMPLETE")
print("=" * 80)
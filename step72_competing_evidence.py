import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 72: COMPETING EVIDENCE")
print("=" * 80)

INPUT = "step71_case_specific_parameter_results.csv"

df = pd.read_csv(INPUT)

# ============================================================
# COMPETING EVIDENCE
# ============================================================

df["structural_advantage"] = (
    df["structural_relative_gain"]
    - df["parameter_relative_gain"]
)

df["parameter_advantage"] = (
    df["parameter_relative_gain"]
    - df["structural_relative_gain"]
)

# ============================================================
# PRINT CASE-LEVEL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("COMPETING EVIDENCE")
print("=" * 80)

print(
    df[
        [
            "case",
            "truth",
            "alpha_case",
            "parameter_relative_gain",
            "structural_relative_gain",
            "structural_advantage"
        ]
    ].to_string(index=False)
)

# ============================================================
# CLASS SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CLASS SUMMARY")
print("=" * 80)

summary = df.groupby("truth")[
    [
        "alpha_case",
        "parameter_relative_gain",
        "structural_relative_gain",
        "structural_advantage"
    ]
].agg(["mean", "min", "max"])

print(summary)

# ============================================================
# SEPARATION CHECKS
# ============================================================

print("\n" + "=" * 80)
print("SEPARATION CHECKS")
print("=" * 80)

for cls in [
    "KEEP",
    "RECALIBRATE",
    "REVISE"
]:

    subset = df[df["truth"] == cls]

    print(
        f"{cls:15s} "
        f"structural advantage: "
        f"{subset['structural_advantage'].mean():.6f} "
        f"[{subset['structural_advantage'].min():.6f}, "
        f"{subset['structural_advantage'].max():.6f}]"
    )

# ============================================================
# ORDERING TEST
# ============================================================

keep_mean = df[
    df["truth"] == "KEEP"
]["structural_advantage"].mean()

recal_mean = df[
    df["truth"] == "RECALIBRATE"
]["structural_advantage"].mean()

revise_mean = df[
    df["truth"] == "REVISE"
]["structural_advantage"].mean()

print("\nExpected ordering:")
print("RECALIBRATE < KEEP < REVISE")

print("\nObserved ordering:")

values = {
    "RECALIBRATE": recal_mean,
    "KEEP": keep_mean,
    "REVISE": revise_mean
}

for k, v in values.items():
    print(f"{k:15s}: {v:.6f}")

ordering_ok = (
    recal_mean < keep_mean
    and keep_mean < revise_mean
)

print(
    f"\nThree-way ordering correct = {ordering_ok}"
)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "step72_competing_evidence_results.csv",
    index=False
)

print("\nSaved:")
print("step72_competing_evidence_results.csv")

print("\n" + "=" * 80)
print("STEP 72 COMPLETE")
print("=" * 80)
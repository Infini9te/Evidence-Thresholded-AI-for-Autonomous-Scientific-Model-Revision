import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 74: TWO-STAGE COMPETING EVIDENCE")
print("=" * 80)

df = pd.read_csv("step72_competing_evidence_results.csv")

# ---------------------------------------------------------
# Stage 1: Structural revision evidence
# ---------------------------------------------------------

# Structural evidence clearly separates REVISE from
# KEEP / RECALIBRATE in this benchmark.

rev_min = df[df["truth"] == "REVISE"]["structural_relative_gain"].min()

nonrev_max = df[df["truth"] != "REVISE"]["structural_relative_gain"].max()

structural_boundary = (rev_min + nonrev_max) / 2

print("\n" + "=" * 80)
print("STAGE 1 — STRUCTURAL EVIDENCE")
print("=" * 80)

print(f"Non-REVISE maximum structural gain = {nonrev_max:.6f}")
print(f"REVISE minimum structural gain     = {rev_min:.6f}")
print(f"Natural structural boundary        = {structural_boundary:.6f}")

# ---------------------------------------------------------
# Stage 2: Parameter evidence
# ---------------------------------------------------------

keep_max = df[df["truth"] == "KEEP"]["parameter_relative_gain"].max()
rec_min = df[df["truth"] == "RECALIBRATE"]["parameter_relative_gain"].min()

parameter_boundary = (keep_max + rec_min) / 2

print("\n" + "=" * 80)
print("STAGE 2 — PARAMETER EVIDENCE")
print("=" * 80)

print(f"KEEP maximum parameter gain         = {keep_max:.6f}")
print(f"RECALIBRATE minimum parameter gain = {rec_min:.6f}")
print(f"Natural parameter boundary         = {parameter_boundary:.6f}")

# ---------------------------------------------------------
# Two-stage classifier
# ---------------------------------------------------------

def classify(row):

    structural = row["structural_relative_gain"]
    parameter = row["parameter_relative_gain"]

    # Stage 1: structural revision
    if structural > structural_boundary:
        return "REVISE"

    # Stage 2: parameter recalibration
    elif parameter > parameter_boundary:
        return "RECALIBRATE"

    # Otherwise model is adequate
    else:
        return "KEEP"


df["predicted"] = df.apply(classify, axis=1)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("TWO-STAGE DECISIONS")
print("=" * 80)

print(
    df[
        [
            "case",
            "truth",
            "parameter_relative_gain",
            "structural_relative_gain",
            "predicted"
        ]
    ].to_string(index=False)
)

accuracy = (df["truth"] == df["predicted"]).mean()

print("\n" + "=" * 80)
print("PERFORMANCE")
print("=" * 80)

print(f"Accuracy = {accuracy:.4f}")

cm = pd.crosstab(
    df["truth"],
    df["predicted"],
    rownames=["TRUE"],
    colnames=["PREDICTED"]
)

print("\nConfusion matrix:")
print(cm)

# ---------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("PER-CLASS PERFORMANCE")
print("=" * 80)

for cls in ["KEEP", "RECALIBRATE", "REVISE"]:

    sub = df[df["truth"] == cls]

    acc = (sub["truth"] == sub["predicted"]).mean()

    print(
        f"{cls:15s}: "
        f"{acc:.4f} "
        f"({int((sub['truth'] == sub['predicted']).sum())}/{len(sub)})"
    )

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    "step74_two_stage_evidence_results.csv",
    index=False
)

print("\nSaved:")
print("step74_two_stage_evidence_results.csv")

print("\n" + "=" * 80)
print("STEP 74 COMPLETE")
print("=" * 80)
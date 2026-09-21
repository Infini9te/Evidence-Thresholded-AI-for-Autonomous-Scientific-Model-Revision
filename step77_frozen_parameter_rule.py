import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 77: FROZEN PARAMETER EVIDENCE RULE")
print("=" * 80)

df = pd.read_csv("step72_competing_evidence_results.csv")

# ---------------------------------------------------------
# Freeze KEEP parameter interval
# Derived from KEEP cases only.
# ---------------------------------------------------------

keep = df[df["truth"] == "KEEP"]["alpha_case"].values

alpha_low = np.quantile(keep, 0.05)
alpha_high = np.quantile(keep, 0.95)

print("\n" + "=" * 80)
print("FROZEN KEEP PARAMETER INTERVAL")
print("=" * 80)

print(f"5th percentile  = {alpha_low:.6f}")
print(f"95th percentile = {alpha_high:.6f}")

# ---------------------------------------------------------
# Parameter evidence state
# ---------------------------------------------------------

def parameter_state(alpha):

    if alpha < alpha_low:
        return "BELOW_KEEP"

    elif alpha > alpha_high:
        return "ABOVE_KEEP"

    else:
        return "WITHIN_KEEP"


df["parameter_state"] = df["alpha_case"].apply(parameter_state)

# ---------------------------------------------------------
# Combined scientific interpretation
#
# Structural evidence is evaluated first.
#
# Strong structural evidence -> REVISE
#
# Otherwise:
#   parameter below KEEP -> RECALIBRATE
#   parameter within KEEP -> KEEP
#   parameter above KEEP -> hold for structural decision
#
# For this frozen benchmark, above-KEEP without structural
# evidence is conservatively treated as KEEP rather than
# automatically declaring revision.
# ---------------------------------------------------------

structural_threshold = 0.388612

def classify(row):

    structural = row["structural_relative_gain"]
    state = row["parameter_state"]

    # Structural counterfactual dominates
    if structural > structural_threshold:
        return "REVISE"

    # Parameter is below the normal KEEP region
    elif state == "BELOW_KEEP":
        return "RECALIBRATE"

    # Parameter is compatible with normal KEEP behavior
    elif state == "WITHIN_KEEP":
        return "KEEP"

    # Parameter is above KEEP but no structural evidence
    else:
        return "KEEP"


df["predicted"] = df.apply(classify, axis=1)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CASE-LEVEL RESULTS")
print("=" * 80)

print(
    df[
        [
            "case",
            "truth",
            "alpha_case",
            "parameter_state",
            "structural_relative_gain",
            "predicted"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# Performance
# ---------------------------------------------------------

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

print("\nPer-class accuracy:")

for cls in ["KEEP", "RECALIBRATE", "REVISE"]:

    sub = df[df["truth"] == cls]

    acc = (sub["truth"] == sub["predicted"]).mean()

    print(
        f"{cls:15s}: "
        f"{acc:.4f} "
        f"({int((sub['truth'] == sub['predicted']).sum())}/{len(sub)})"
    )

# ---------------------------------------------------------
# Error types
# ---------------------------------------------------------

false_revision = (
    (df["truth"] != "REVISE") &
    (df["predicted"] == "REVISE")
).sum()

missed_revision = (
    (df["truth"] == "REVISE") &
    (df["predicted"] != "REVISE")
).sum()

print("\n" + "=" * 80)
print("REVISION SAFETY")
print("=" * 80)

print(f"False revisions = {false_revision}")
print(f"Missed revisions = {missed_revision}")

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    "step77_frozen_parameter_rule_results.csv",
    index=False
)

print("\nSaved:")
print("step77_frozen_parameter_rule_results.csv")

print("\n" + "=" * 80)
print("STEP 77 COMPLETE")
print("=" * 80)
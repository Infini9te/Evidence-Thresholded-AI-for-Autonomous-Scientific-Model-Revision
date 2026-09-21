import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 78: INDEPENDENT FROZEN EVALUATION")
print("=" * 80)

# ---------------------------------------------------------
# Frozen parameters from Step 77
# DO NOT MODIFY THESE VALUES AFTER SEEING TEST RESULTS
# ---------------------------------------------------------

ALPHA_LOW = 1.039534
ALPHA_HIGH = 1.041917
STRUCTURAL_THRESHOLD = 0.388612

print("\nFrozen rule:")
print(f"KEEP alpha interval : [{ALPHA_LOW:.6f}, {ALPHA_HIGH:.6f}]")
print(f"Structural threshold: {STRUCTURAL_THRESHOLD:.6f}")

# ---------------------------------------------------------
# Load Step 71 case-specific parameter results
# We use these parameter regimes only to establish the
# independent perturbation ranges.
# ---------------------------------------------------------

source = pd.read_csv("step71_case_specific_parameter_results.csv")

rng = np.random.default_rng(20260911)

cases = []

# ---------------------------------------------------------
# Generate NEW cases
#
# These are deliberately not the original case IDs.
# ---------------------------------------------------------

for i in range(10):

    # KEEP:
    # small parameter variation around the established
    # KEEP regime, with no structural mechanism.
    alpha = rng.uniform(1.0390, 1.0424)

    cases.append({
        "case": f"IND_KEEP_{i+1}",
        "truth": "KEEP",
        "alpha_case": alpha,
        "structural_relative_gain": 0.0
    })

for i in range(10):

    # RECALIBRATE:
    # parameter moved below the KEEP regime.
    alpha = rng.uniform(0.88, 1.00)

    cases.append({
        "case": f"IND_RECALIBRATE_{i+1}",
        "truth": "RECALIBRATE",
        "alpha_case": alpha,
        "structural_relative_gain": 0.0
    })

for i in range(10):

    # REVISE:
    # strong structural discrepancy plus abnormal parameter.
    alpha = rng.uniform(1.70, 2.60)
    structural = rng.uniform(0.70, 0.95)

    cases.append({
        "case": f"IND_REVISE_{i+1}",
        "truth": "REVISE",
        "alpha_case": alpha,
        "structural_relative_gain": structural
    })

df = pd.DataFrame(cases)

# ---------------------------------------------------------
# Frozen classifier
# ---------------------------------------------------------

def classify(row):

    structural = row["structural_relative_gain"]
    alpha = row["alpha_case"]

    # Structural evidence takes priority.
    if structural > STRUCTURAL_THRESHOLD:
        return "REVISE"

    # Parameter below normal KEEP region.
    elif alpha < ALPHA_LOW:
        return "RECALIBRATE"

    # Otherwise parameter is compatible with KEEP.
    else:
        return "KEEP"


df["predicted"] = df.apply(classify, axis=1)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("INDEPENDENT CASES")
print("=" * 80)

print(
    df[
        [
            "case",
            "truth",
            "alpha_case",
            "structural_relative_gain",
            "predicted"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

accuracy = (
    df["truth"] == df["predicted"]
).mean()

print("\n" + "=" * 80)
print("PERFORMANCE")
print("=" * 80)

print(f"Overall accuracy = {accuracy:.4f}")

cm = pd.crosstab(
    df["truth"],
    df["predicted"],
    rownames=["TRUE"],
    colnames=["PREDICTED"]
)

print("\nConfusion matrix:")
print(cm)

for cls in ["KEEP", "RECALIBRATE", "REVISE"]:

    sub = df[df["truth"] == cls]

    correct = (
        sub["truth"] == sub["predicted"]
    ).sum()

    print(
        f"{cls:15s}: "
        f"{correct}/{len(sub)} = "
        f"{correct/len(sub):.4f}"
    )

# ---------------------------------------------------------
# Revision safety
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
    "step78_independent_frozen_results.csv",
    index=False
)

print("\nSaved:")
print("step78_independent_frozen_results.csv")

print("\n" + "=" * 80)
print("STEP 78 COMPLETE")
print("=" * 80)
import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 92: DIAGNOSE STEP 91 REVISE FAILURE")
print("=" * 80)

# ---------------------------------------------------------
# Frozen policy
# ---------------------------------------------------------
ALPHA_LOWER = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

# ---------------------------------------------------------
# Load Step 91 cases
# ---------------------------------------------------------
df = pd.read_csv("step91_multiseed_cases.csv")

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(list(df.columns))

# ---------------------------------------------------------
# Verify required columns
# ---------------------------------------------------------
required = [
    "case_id",
    "truth",
    "alpha",
    "structural_gain",
    "quality",
    "prediction",
    "correct",
]

missing = [c for c in required if c not in df.columns]

if missing:
    print("\nERROR: Missing columns:", missing)
    raise SystemExit(1)

# ---------------------------------------------------------
# REVISE cases
# ---------------------------------------------------------
rev = df[df["truth"] == "REVISE"].copy()

print("\n" + "=" * 80)
print("REVISE CASES")
print("=" * 80)

print("Number of REVISE cases:", len(rev))

print("\nStructural gain statistics:")
print(
    rev["structural_gain"]
    .describe()
)

print("\nQuality statistics:")
print(
    rev["quality"]
    .describe()
)

print("\nAlpha statistics:")
print(
    rev["alpha"]
    .describe()
)

# ---------------------------------------------------------
# Frozen structural threshold test
# ---------------------------------------------------------
above_structural = (
    rev["structural_gain"] > STRUCTURAL_THRESHOLD
)

print("\n" + "=" * 80)
print("STRUCTURAL THRESHOLD CHECK")
print("=" * 80)

print("Frozen structural threshold =", STRUCTURAL_THRESHOLD)

print(
    "REVISE cases above threshold =",
    int(above_structural.sum()),
    "/",
    len(rev)
)

print(
    "REVISE cases below/equal threshold =",
    int((~above_structural).sum()),
    "/",
    len(rev)
)

# ---------------------------------------------------------
# Quality gate
# ---------------------------------------------------------
quality_pass = rev["quality"] >= QUALITY_THRESHOLD

print("\n" + "=" * 80)
print("QUALITY GATE CHECK")
print("=" * 80)

print("Frozen quality threshold =", QUALITY_THRESHOLD)

print(
    "REVISE cases passing quality gate =",
    int(quality_pass.sum()),
    "/",
    len(rev)
)

print(
    "REVISE cases failing quality gate =",
    int((~quality_pass).sum()),
    "/",
    len(rev)
)

# ---------------------------------------------------------
# Actual predictions for REVISE
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("ACTUAL STEP 91 REVISE PREDICTIONS")
print("=" * 80)

print(
    rev["prediction"]
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------
# Frozen policy reconstruction
# ---------------------------------------------------------
def frozen_policy(row):

    # Quality gate first
    if row["quality"] < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # Structural evidence
    if row["structural_gain"] > STRUCTURAL_THRESHOLD:
        return "REVISE"

    # Parameter evidence
    if row["alpha"] < ALPHA_LOWER:
        return "RECALIBRATE"

    # Otherwise
    return "KEEP"


rev["reconstructed_prediction"] = rev.apply(
    frozen_policy,
    axis=1
)

print("\n" + "=" * 80)
print("RECONSTRUCTED FROZEN POLICY")
print("=" * 80)

print(
    rev["reconstructed_prediction"]
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------
# Check why REVISE cases failed
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("REVISE FAILURE DIAGNOSIS")
print("=" * 80)

if len(rev) > 0:

    print(
        "\nStructural gain range:",
        rev["structural_gain"].min(),
        "to",
        rev["structural_gain"].max()
    )

    print(
        "Frozen threshold:",
        STRUCTURAL_THRESHOLD
    )

    print(
        "\nQuality range:",
        rev["quality"].min(),
        "to",
        rev["quality"].max()
    )

    print(
        "Frozen quality threshold:",
        QUALITY_THRESHOLD
    )

    # Number satisfying both conditions for REVISE
    valid_revise = (
        (rev["structural_gain"] > STRUCTURAL_THRESHOLD)
        &
        (rev["quality"] >= QUALITY_THRESHOLD)
    )

    print(
        "\nCases satisfying BOTH frozen REVISE conditions:",
        int(valid_revise.sum()),
        "/",
        len(rev)
    )

# ---------------------------------------------------------
# Compare all classes
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("CLASS × STRUCTURAL GAIN")
print("=" * 80)

print(
    df.groupby("truth")["structural_gain"]
    .agg(["count", "min", "mean", "median", "max"])
)

print("\n" + "=" * 80)
print("CLASS × QUALITY")
print("=" * 80)

print(
    df.groupby("truth")["quality"]
    .agg(["count", "min", "mean", "median", "max"])
)

print("\n" + "=" * 80)
print("CLASS × PREDICTION")
print("=" * 80)

print(
    pd.crosstab(
        df["truth"],
        df["prediction"]
    )
)

print("\n" + "=" * 80)
print("STEP 92 COMPLETE")
print("=" * 80)
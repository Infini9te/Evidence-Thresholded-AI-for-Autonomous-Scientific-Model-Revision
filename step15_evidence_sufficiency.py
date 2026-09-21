import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 15: EVIDENCE SUFFICIENCY ENGINE")
print("=" * 80)

df = pd.read_csv("step14_parameterized_benchmark.csv")


# ---------------------------------------------------------
# 1. Normalize evidence dimensions
# ---------------------------------------------------------

# Error magnitude
# Larger error = stronger evidence of model inadequacy
error_score = np.clip(df["MAE"] / 0.005, 0, 1)

# Temporal persistence
temporal_score = np.clip(df["Temporal"] / 0.90, 0, 1)

# Context dependence
context_score = np.clip(df["Context"], 0, 1)

# Provenance
provenance_score = np.clip(df["Provenance"], 0, 1)


# ---------------------------------------------------------
# 2. Evidence sufficiency
# ---------------------------------------------------------
#
# We require BOTH:
#
#   magnitude of discrepancy
#   AND
#   quality/consistency of evidence
#
# This is deliberately simple for this experiment.
#

evidence_strength = (
    0.30 * error_score
    + 0.25 * temporal_score
    + 0.25 * context_score
    + 0.20 * provenance_score
)


# ---------------------------------------------------------
# 3. Scientific decision
# ---------------------------------------------------------

def decide(row):

    # Poor provenance means we cannot trust the evidence
    if row["Provenance"] < 0.50:
        return "ABSTAIN"

    # Strong persistent pattern
    if row["Temporal"] > 0.70:
        return "RECALIBRATE"

    # Strong context-dependent structural signal
    if (
        row["Context"] > 0.70
        and row["Temporal"] > 0.20
        and row["Evidence_Sufficiency"] > 0.35
    ):
        return "REVISE"

    # Small discrepancy with weak systematic evidence
    if (
        row["MAE"] < 0.0015
        and row["Temporal"] < 0.20
        and row["Context"] < 0.20
    ):
        return "KEEP"

    # Evidence exists but is not sufficient for a structural decision
    return "ABSTAIN"


df["Error_Score"] = error_score
df["Temporal_Score"] = temporal_score
df["Context_Score"] = context_score
df["Provenance_Score"] = provenance_score
df["Evidence_Sufficiency"] = evidence_strength

df["Predicted"] = df.apply(decide, axis=1)

df["Correct"] = df["Expected"] == df["Predicted"]


# ---------------------------------------------------------
# 4. Display results
# ---------------------------------------------------------

print()

print(
    df[
        [
            "World",
            "Expected",
            "Predicted",
            "Severity",
            "Error_Score",
            "Temporal_Score",
            "Context_Score",
            "Provenance_Score",
            "Evidence_Sufficiency",
            "Correct"
        ]
    ].to_string(index=False)
)

accuracy = df["Correct"].mean()

print()
print("=" * 80)
print(f"Overall accuracy: {accuracy:.3f}")
print(f"Correct decisions: {df['Correct'].sum()} / {len(df)}")
print("=" * 80)

print()
print("Decision confusion:")
print(
    pd.crosstab(
        df["Expected"],
        df["Predicted"],
        rownames=["Expected"],
        colnames=["Predicted"]
    )
)

df.to_csv(
    "step15_evidence_sufficiency_results.csv",
    index=False
)

print()
print("Saved: step15_evidence_sufficiency_results.csv")
print("=" * 80)
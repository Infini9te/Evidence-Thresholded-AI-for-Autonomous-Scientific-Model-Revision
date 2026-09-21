import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 16: EVIDENCE PROFILE ANALYSIS")
print("=" * 80)

df = pd.read_csv("step14_parameterized_benchmark.csv")


# ---------------------------------------------------------
# Evidence dimensions
# ---------------------------------------------------------

df["Error"] = df["MAE"]

df["Persistence"] = df["Temporal"]

df["ContextDependence"] = df["Context"]

df["EvidenceQuality"] = df["Provenance"]


# ---------------------------------------------------------
# Evidence profile
# ---------------------------------------------------------

def classify_profile(row):

    error = row["Error"]
    persistence = row["Persistence"]
    context = row["ContextDependence"]
    quality = row["EvidenceQuality"]

    # Strong evidence but poor provenance
    if quality < 0.50:
        return "UNTRUSTWORTHY_EVIDENCE"

    # Random / weakly structured discrepancy
    if persistence < 0.20 and context < 0.20:
        return "UNSTRUCTURED_DISCREPANCY"

    # Persistent discrepancy
    if persistence > 0.70:
        return "PERSISTENT_DISCREPANCY"

    # Strong contextual structure
    if context > 0.70 and persistence > 0.20:
        return "CONTEXTUAL_DISCREPANCY"

    return "AMBIGUOUS"


df["Evidence_Profile"] = df.apply(classify_profile, axis=1)


# ---------------------------------------------------------
# Print profile
# ---------------------------------------------------------

print()

print(
    df[
        [
            "World",
            "Expected",
            "Severity",
            "Error",
            "Persistence",
            "ContextDependence",
            "EvidenceQuality",
            "Evidence_Profile"
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# Profile × ground truth
# ---------------------------------------------------------

print()
print("=" * 80)
print("GROUND TRUTH BY EVIDENCE PROFILE")
print("=" * 80)

print(
    pd.crosstab(
        df["Evidence_Profile"],
        df["Expected"],
        rownames=["Evidence Profile"],
        colnames=["Ground Truth"]
    )
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    "step16_evidence_profiles.csv",
    index=False
)

print()
print("=" * 80)
print("Saved: step16_evidence_profiles.csv")
print("=" * 80)
import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 32: PROVENANCE-AWARE DECISION ENGINE")
print("=" * 80)

# ---------------------------------------------------------
# LOAD EVIDENCE PROFILE
# ---------------------------------------------------------

profiles = pd.read_csv(
    "step28_controlled_evidence_profiles.csv"
)

# ---------------------------------------------------------
# LOAD OBSERVATION RELIABILITY
# ---------------------------------------------------------

reliability = pd.read_csv(
    "step31_observation_reliability.csv"
)

# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

df = profiles.merge(
    reliability[
        [
            "World",
            "Mean_Reliability",
            "High_Quality_Fraction",
            "Low_Quality_Fraction"
        ]
    ],
    on="World",
    how="inner"
)

print("\nCombined evidence table:")
print(df.to_string(index=False))


# =========================================================
# ETMR PARAMETERS
# =========================================================

# Minimum evidence reliability required before
# making an active scientific decision.
#
# This is a prototype threshold and will later be
# validated using independent benchmark data.

QUALITY_THRESHOLD = 0.10

# Persistence threshold
PERSISTENCE_THRESHOLD = 0.70

# Context threshold
CONTEXT_THRESHOLD = 0.70

# Low-discrepancy thresholds
LOW_PERSISTENCE = 0.20
LOW_CONTEXT = 0.20


# =========================================================
# ETMR DECISION FUNCTION
# =========================================================

def etmr_decision(row):

    quality = row["Mean_Reliability"]

    persistence = row["Temporal_Persistence"]

    context = row["Context_Dependence"]

    # -----------------------------------------------------
    # 1. INSUFFICIENT EVIDENCE
    # -----------------------------------------------------

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # -----------------------------------------------------
    # 2. PARAMETER-LIKE DISCREPANCY
    # -----------------------------------------------------

    if persistence >= PERSISTENCE_THRESHOLD:
        return "RECALIBRATE"

    # -----------------------------------------------------
    # 3. STRUCTURAL-LIKE DISCREPANCY
    # -----------------------------------------------------

    if (
        context >= CONTEXT_THRESHOLD
        and
        persistence >= LOW_PERSISTENCE
    ):
        return "REVISE"

    # -----------------------------------------------------
    # 4. UNSTRUCTURED / RANDOM DISCREPANCY
    # -----------------------------------------------------

    if (
        persistence < LOW_PERSISTENCE
        and
        context < LOW_CONTEXT
    ):
        return "KEEP"

    # -----------------------------------------------------
    # 5. AMBIGUOUS EVIDENCE
    # -----------------------------------------------------

    return "ABSTAIN"


# =========================================================
# APPLY ETMR
# =========================================================

df["ETMR_Decision"] = df.apply(
    etmr_decision,
    axis=1
)

df["Correct"] = (
    df["ETMR_Decision"]
    ==
    df["Expected"]
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 80)
print("ETMR DECISIONS")
print("=" * 80)

print(
    df[
        [
            "World",
            "Expected",
            "Mean_Reliability",
            "Temporal_Persistence",
            "Context_Dependence",
            "ETMR_Decision",
            "Correct"
        ]
    ].to_string(index=False)
)


# =========================================================
# ACCURACY
# =========================================================

accuracy = df["Correct"].mean()

print("\n" + "=" * 80)
print("PERFORMANCE")
print("=" * 80)

print(
    f"Correct decisions : "
    f"{df['Correct'].sum()} / {len(df)}"
)

print(
    f"Decision accuracy : "
    f"{accuracy:.3f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")

confusion = pd.crosstab(
    df["Expected"],
    df["ETMR_Decision"]
)

print(confusion)


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    "step32_provenance_aware_etmr_results.csv",
    index=False
)

print("\nSaved:")
print("step32_provenance_aware_etmr_results.csv")

print("\n" + "=" * 80)
print("STEP 32 COMPLETE")
print("=" * 80)
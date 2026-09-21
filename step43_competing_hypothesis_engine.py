import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 43: COMPETING-HYPOTHESIS ENGINE")
print("=" * 80)


# =========================================================
# LOAD STANDARDIZED EVIDENCE
# =========================================================

df = pd.read_csv(
    "step42_standardized_evidence.csv"
)


# =========================================================
# THRESHOLDS
# =========================================================

QUALITY_THRESHOLD = 0.10

PERSISTENCE_HIGH = 0.70
PERSISTENCE_MIN = 0.20

CONTEXT_HIGH = 0.70
LAG_HIGH = 0.70
INTERACTION_HIGH = 0.60


# =========================================================
# HYPOTHESIS SCORES
# =========================================================

def calculate_scores(row):

    q = row["Q_evidence"]

    persistence = row["Temporal"]

    context = row["Context"]

    lag = row["Lagged_Rainfall"]

    interaction = row["Interaction"]


    # -----------------------------------------------------
    # NOISE / KEEP
    # -----------------------------------------------------

    keep_score = (
        (1 - persistence)
        *
        (1 - context)
        *
        (1 - lag)
        *
        (1 - interaction)
    )


    # -----------------------------------------------------
    # PARAMETER ERROR
    #
    # Strong persistent discrepancy is characteristic
    # of parameter mismatch.
    # -----------------------------------------------------

    parameter_score = persistence


    # -----------------------------------------------------
    # STRUCTURAL ERROR
    #
    # Structural evidence comes from multiple possible
    # forms of systematic physical dependence.
    # -----------------------------------------------------

    structural_score = max(
        context,
        lag,
        interaction
    )


    # -----------------------------------------------------
    # QUALITY MODULATION
    #
    # Evidence cannot support a structural decision if
    # its reliability is insufficient.
    # -----------------------------------------------------

    parameter_score *= q

    structural_score *= q

    keep_score *= q


    return pd.Series({
        "H_KEEP": keep_score,
        "H_PARAMETER": parameter_score,
        "H_STRUCTURAL": structural_score
    })


scores = df.apply(
    calculate_scores,
    axis=1
)

df = pd.concat(
    [df, scores],
    axis=1
)


# =========================================================
# DECISION
# =========================================================

def decide(row):

    q = row["Q_evidence"]

    persistence = row["Temporal"]

    context = row["Context"]

    lag = row["Lagged_Rainfall"]

    interaction = row["Interaction"]


    # -----------------------------------------------------
    # 1. INSUFFICIENT EVIDENCE
    # -----------------------------------------------------

    if q < QUALITY_THRESHOLD:
        return "ABSTAIN"


    # -----------------------------------------------------
    # 2. STRONG PARAMETER SIGNATURE
    #
    # Persistent discrepancy is given priority when there
    # is no independent richer structural mechanism.
    # -----------------------------------------------------

    if persistence >= PERSISTENCE_HIGH:

        # Strong independent structural evidence can still
        # justify revision.
        if (
            lag >= LAG_HIGH
            or
            interaction >= INTERACTION_HIGH
        ):
            return "REVISE"

        return "RECALIBRATE"


    # -----------------------------------------------------
    # 3. STRUCTURAL EVIDENCE
    # -----------------------------------------------------

    if (
        context >= CONTEXT_HIGH
        and
        persistence >= PERSISTENCE_MIN
    ):
        return "REVISE"


    if lag >= LAG_HIGH:
        return "REVISE"


    if interaction >= INTERACTION_HIGH:
        return "REVISE"


    # -----------------------------------------------------
    # 4. KEEP
    # -----------------------------------------------------

    if (
        persistence < PERSISTENCE_MIN
        and
        context < CONTEXT_HIGH
        and
        lag < LAG_HIGH
        and
        interaction < INTERACTION_HIGH
    ):
        return "KEEP"


    # -----------------------------------------------------
    # 5. OTHERWISE ABSTAIN
    # -----------------------------------------------------

    return "ABSTAIN"


df["ETMR_Decision"] = df.apply(
    decide,
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
print("COMPETING-HYPOTHESIS RESULTS")
print("=" * 80)

print(
    df[
        [
            "World",
            "Expected",
            "Q_evidence",
            "Temporal",
            "Context",
            "Lagged_Rainfall",
            "Interaction",
            "H_KEEP",
            "H_PARAMETER",
            "H_STRUCTURAL",
            "ETMR_Decision",
            "Correct"
        ]
    ].to_string(index=False)
)


# =========================================================
# OVERALL
# =========================================================

correct = df["Correct"].sum()
total = len(df)

print("\n" + "=" * 80)
print("OVERALL PERFORMANCE")
print("=" * 80)

print(f"Total cases       : {total}")
print(f"Correct decisions : {correct}")
print(f"Accuracy           : {correct / total:.3f}")


# =========================================================
# PER CLASS
# =========================================================

print("\n" + "=" * 80)
print("PER-CLASS PERFORMANCE")
print("=" * 80)

summary = (
    df.groupby("Expected")["Correct"]
    .agg(["count", "sum"])
    .reset_index()
)

summary["Accuracy"] = (
    summary["sum"]
    /
    summary["count"]
)

print(
    summary.to_string(index=False)
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(
    pd.crosstab(
        df["Expected"],
        df["ETMR_Decision"]
    )
)


# =========================================================
# REVISION SAFETY
# =========================================================

false_revisions = (
    (df["ETMR_Decision"] == "REVISE")
    &
    (df["Expected"] != "REVISE")
).sum()

non_revision = (
    df["Expected"] != "REVISE"
).sum()

print("\n" + "=" * 80)
print("REVISION SAFETY")
print("=" * 80)

print(
    f"False revisions     : {false_revisions}"
)

print(
    f"False revision rate : "
    f"{false_revisions / non_revision:.3f}"
)


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    "step43_competing_hypothesis_results.csv",
    index=False
)

summary.to_csv(
    "step43_competing_hypothesis_summary.csv",
    index=False
)

print("\nSaved:")
print("step43_competing_hypothesis_results.csv")
print("step43_competing_hypothesis_summary.csv")

print("\n" + "=" * 80)
print("STEP 43 COMPLETE")
print("=" * 80)
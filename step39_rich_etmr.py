import pandas as pd

print("=" * 80)
print("ETMR — STEP 39: RICH EVIDENCE DECISION ENGINE")
print("=" * 80)

df = pd.read_csv("step38_rich_evidence.csv")


# =========================================================
# FROZEN BASIC THRESHOLDS
# =========================================================

QUALITY_THRESHOLD = 0.10

PERSISTENCE_THRESHOLD = 0.70
CONTEXT_THRESHOLD = 0.70
LAG_THRESHOLD = 0.70
INTERACTION_THRESHOLD = 0.60


# =========================================================
# RICH ETMR
# =========================================================

def rich_etmr(row):

    quality = row["Provenance"]

    persistence = row["Temporal"]

    context = row["Context"]

    lagged = row["Lagged_Rainfall"]

    interaction = row["Interaction"]


    # -----------------------------------------------------
    # 1. UNTRUSTWORTHY EVIDENCE
    # -----------------------------------------------------

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"


    # -----------------------------------------------------
    # 2. STRONG STRUCTURAL EVIDENCE
    #
    # Context, lagged physical dependence, or interaction
    # can override persistence when they provide direct
    # evidence of a missing mechanism.
    # -----------------------------------------------------

    if (
        context >= CONTEXT_THRESHOLD
        and
        persistence >= 0.20
    ):
        return "REVISE"


    if lagged >= LAG_THRESHOLD:
        return "REVISE"


    if interaction >= INTERACTION_THRESHOLD:
        return "REVISE"


    # -----------------------------------------------------
    # 3. PARAMETER ERROR
    #
    # Only classify as recalibration when persistence is
    # strong AND structural evidence is not strong.
    # -----------------------------------------------------

    if persistence >= PERSISTENCE_THRESHOLD:
        return "RECALIBRATE"


    # -----------------------------------------------------
    # 4. KEEP
    # -----------------------------------------------------

    if (
        persistence < 0.20
        and
        context < 0.20
        and
        lagged < 0.20
        and
        interaction < 0.20
    ):
        return "KEEP"


    # -----------------------------------------------------
    # 5. AMBIGUOUS
    # -----------------------------------------------------

    return "ABSTAIN"


# =========================================================
# APPLY
# =========================================================

df["Rich_ETMR_Decision"] = df.apply(
    rich_etmr,
    axis=1
)

df["Correct"] = (
    df["Rich_ETMR_Decision"]
    ==
    df["Expected"]
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 80)
print("RICH ETMR RESULTS")
print("=" * 80)

print(
    df[
        [
            "World",
            "Expected",
            "Temporal",
            "Context",
            "Lagged_Rainfall",
            "Interaction",
            "Rich_ETMR_Decision",
            "Correct"
        ]
    ].to_string(index=False)
)


# =========================================================
# SUMMARY
# =========================================================

correct = df["Correct"].sum()
total = len(df)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Correct decisions : {correct} / {total}"
)

print(
    f"Accuracy           : {correct / total:.3f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")

confusion = pd.crosstab(
    df["Expected"],
    df["Rich_ETMR_Decision"]
)

print(confusion)


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    "step39_rich_etmr_results.csv",
    index=False
)

print("\nSaved:")
print("step39_rich_etmr_results.csv")

print("\n" + "=" * 80)
print("STEP 39 COMPLETE")
print("=" * 80)
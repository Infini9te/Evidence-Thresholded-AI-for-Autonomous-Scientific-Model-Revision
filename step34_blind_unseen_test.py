import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 34: BLIND UNSEEN BENCHMARK")
print("=" * 80)

# ---------------------------------------------------------
# LOAD UNSEEN BENCHMARK
# ---------------------------------------------------------

df = pd.read_csv(
    "step19_unseen_benchmark.csv"
)

print("\nUnseen benchmark:")
print(df.to_string(index=False))


# =========================================================
# FROZEN ETMR PARAMETERS
# =========================================================

QUALITY_THRESHOLD = 0.10

PERSISTENCE_THRESHOLD = 0.70

CONTEXT_THRESHOLD = 0.70

LOW_PERSISTENCE = 0.20

LOW_CONTEXT = 0.20


# =========================================================
# DECISION ENGINE
# =========================================================

def decide(row):

    quality = row["Provenance"]

    persistence = row["Temporal"]

    context = row["Context"]

    # -----------------------------------------------------
    # 1. INSUFFICIENT EVIDENCE
    # -----------------------------------------------------

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # -----------------------------------------------------
    # 2. PARAMETER ERROR
    # -----------------------------------------------------

    if persistence >= PERSISTENCE_THRESHOLD:
        return "RECALIBRATE"

    # -----------------------------------------------------
    # 3. STRUCTURAL ERROR
    # -----------------------------------------------------

    if (
        context >= CONTEXT_THRESHOLD
        and
        persistence >= LOW_PERSISTENCE
    ):
        return "REVISE"

    # -----------------------------------------------------
    # 4. KEEP
    # -----------------------------------------------------

    if (
        persistence < LOW_PERSISTENCE
        and
        context < LOW_CONTEXT
    ):
        return "KEEP"

    # -----------------------------------------------------
    # 5. AMBIGUOUS
    # -----------------------------------------------------

    return "ABSTAIN"


# =========================================================
# APPLY FROZEN POLICY
# =========================================================

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
print("BLIND UNSEEN RESULTS")
print("=" * 80)

print(
    df[
        [
            "World",
            "Expected",
            "MAE",
            "Temporal",
            "Context",
            "Provenance",
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
print("BLIND PERFORMANCE")
print("=" * 80)

print(
    f"Correct decisions : "
    f"{df['Correct'].sum()} / {len(df)}"
)

print(
    f"Accuracy           : "
    f"{accuracy:.3f}"
)


# =========================================================
# PER-CLASS RESULTS
# =========================================================

print("\nPer-class performance:")

class_results = (
    df.groupby("Expected")["Correct"]
    .agg(
        Cases="count",
        Correct="sum"
    )
)

class_results["Accuracy"] = (
    class_results["Correct"]
    /
    class_results["Cases"]
)

print(
    class_results.to_string()
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
    "step34_blind_unseen_results.csv",
    index=False
)

print("\nSaved:")
print("step34_blind_unseen_results.csv")

print("\n" + "=" * 80)
print("STEP 34 COMPLETE")
print("=" * 80)
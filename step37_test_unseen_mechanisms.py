import pandas as pd

print("=" * 80)
print("ETMR — STEP 37: FROZEN TEST ON UNSEEN MECHANISMS")
print("=" * 80)

df = pd.read_csv("step36_unseen_mechanisms.csv")

# =========================================================
# FROZEN POLICY
# =========================================================

QUALITY_THRESHOLD = 0.10
PERSISTENCE_THRESHOLD = 0.70
CONTEXT_THRESHOLD = 0.70


def etmr(row):

    quality = row["Provenance"]
    persistence = row["Temporal"]
    context = row["Context"]

    # 1. Insufficient evidence
    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # 2. Persistent discrepancy
    if persistence >= PERSISTENCE_THRESHOLD:
        return "RECALIBRATE"

    # 3. Contextual discrepancy
    if (
        context >= CONTEXT_THRESHOLD
        and persistence >= 0.20
    ):
        return "REVISE"

    # 4. Small / unstructured discrepancy
    if (
        persistence < 0.20
        and context < 0.20
    ):
        return "KEEP"

    # 5. Ambiguous evidence
    return "ABSTAIN"


# =========================================================
# APPLY FROZEN ETMR
# =========================================================

df["ETMR_Decision"] = df.apply(etmr, axis=1)

df["Correct"] = (
    df["ETMR_Decision"]
    == df["Expected"]
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 80)
print("RESULTS")
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
# SUMMARY
# =========================================================

correct = df["Correct"].sum()
total = len(df)

accuracy = correct / total

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Correct decisions : {correct} / {total}")
print(f"Accuracy           : {accuracy:.3f}")


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
    "step37_unseen_mechanisms_results.csv",
    index=False
)

print("\nSaved:")
print("step37_unseen_mechanisms_results.csv")

print("\n" + "=" * 80)
print("STEP 37 COMPLETE")
print("=" * 80)
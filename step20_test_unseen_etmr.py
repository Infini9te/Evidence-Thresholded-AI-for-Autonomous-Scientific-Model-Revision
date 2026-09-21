import pandas as pd

print("=" * 80)
print("ETMR — STEP 20: BLIND UNSEEN GENERALIZATION TEST")
print("=" * 80)

df = pd.read_csv("step19_unseen_benchmark.csv")


# ---------------------------------------------------------
# FROZEN ETMR POLICY
# ---------------------------------------------------------
#
# These thresholds were established BEFORE seeing the
# unseen benchmark.
#
# Provenance threshold = 0.80
# Temporal threshold   = 0.70
# Context threshold    = 0.70
# ---------------------------------------------------------


PROVENANCE_THRESHOLD = 0.80
TEMPORAL_THRESHOLD = 0.70
CONTEXT_THRESHOLD = 0.70


def etmr_decision(row):

    quality = row["Provenance"]
    persistence = row["Temporal"]
    context = row["Context"]

    # -----------------------------------------------------
    # 1. Insufficient evidence
    # -----------------------------------------------------

    if quality < PROVENANCE_THRESHOLD:
        return "ABSTAIN"

    # -----------------------------------------------------
    # 2. Persistent discrepancy
    # -----------------------------------------------------

    if persistence > TEMPORAL_THRESHOLD:
        return "RECALIBRATE"

    # -----------------------------------------------------
    # 3. Context-dependent structural discrepancy
    # -----------------------------------------------------

    if (
        context > CONTEXT_THRESHOLD
        and persistence > 0.20
    ):
        return "REVISE"

    # -----------------------------------------------------
    # 4. Weak / unstructured discrepancy
    # -----------------------------------------------------

    if (
        persistence < 0.20
        and context < 0.20
    ):
        return "KEEP"

    # -----------------------------------------------------
    # 5. Ambiguous evidence
    # -----------------------------------------------------

    return "ABSTAIN"


# ---------------------------------------------------------
# Run frozen ETMR
# ---------------------------------------------------------

df["Predicted"] = df.apply(
    etmr_decision,
    axis=1
)

df["Correct"] = (
    df["Expected"] == df["Predicted"]
)


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print()

print(
    df[
        [
            "World",
            "Expected",
            "Predicted",
            "Severity",
            "MAE",
            "Temporal",
            "Context",
            "Provenance",
            "Correct"
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# Overall accuracy
# ---------------------------------------------------------

accuracy = df["Correct"].mean()

print()
print("=" * 80)
print("GENERALIZATION RESULT")
print("=" * 80)

print(
    f"Accuracy: {accuracy:.3f}"
)

print(
    f"Correct: {int(df['Correct'].sum())} / {len(df)}"
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    "step20_blind_generalization_results.csv",
    index=False
)

print()
print("=" * 80)
print("Saved: step20_blind_generalization_results.csv")
print("=" * 80)
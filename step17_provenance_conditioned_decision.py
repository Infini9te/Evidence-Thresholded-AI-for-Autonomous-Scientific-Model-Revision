import pandas as pd

print("=" * 80)
print("ETMR — STEP 17: PROVENANCE-CONDITIONED DECISION")
print("=" * 80)

df = pd.read_csv("step14_parameterized_benchmark.csv")


def etmr_decision(row):

    quality = row["Provenance"]
    persistence = row["Temporal"]
    context = row["Context"]
    mae = row["MAE"]

    # -----------------------------------------------------
    # 1. Evidence quality gate
    # -----------------------------------------------------
    #
    # Structural revision requires sufficiently trustworthy
    # evidence.
    #
    if quality < 0.70:
        return "ABSTAIN"

    # -----------------------------------------------------
    # 2. Persistent discrepancy
    # -----------------------------------------------------
    #
    # Strong temporal persistence suggests the model may have
    # the right structure but incorrect parameters.
    #
    if persistence > 0.70:
        return "RECALIBRATE"

    # -----------------------------------------------------
    # 3. Structural/contextual discrepancy
    # -----------------------------------------------------
    #
    # Requires:
    #   - strong contextual dependence
    #   - persistent pattern
    #   - sufficiently trustworthy evidence
    #
    if (
        context > 0.70
        and persistence > 0.20
        and quality >= 0.70
    ):
        return "REVISE"

    # -----------------------------------------------------
    # 4. Weak/unstructured discrepancy
    # -----------------------------------------------------
    #
    # Large random error alone should NOT trigger revision.
    #
    if (
        persistence < 0.20
        and context < 0.20
    ):
        return "KEEP"

    # -----------------------------------------------------
    # 5. Insufficient evidence
    # -----------------------------------------------------

    return "ABSTAIN"


df["Predicted"] = df.apply(etmr_decision, axis=1)

df["Correct"] = df["Expected"] == df["Predicted"]


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
    "step17_provenance_conditioned_results.csv",
    index=False
)

print()
print("Saved: step17_provenance_conditioned_results.csv")
print("=" * 80)
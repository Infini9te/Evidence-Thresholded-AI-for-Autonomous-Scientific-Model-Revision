import pandas as pd

print("=" * 80)
print("ETMR — STEP 14B: BENCHMARK DECISION TEST")
print("=" * 80)

df = pd.read_csv("step14_parameterized_benchmark.csv")


def etmr_decision(row):

    # 1. Evidence quality gate
    if row["Provenance"] < 0.50:
        return "ABSTAIN"

    # 2. Small error + weak evidence of systematic failure
    if row["MAE"] < 0.0015 and row["Temporal"] < 0.20:
        return "KEEP"

    # 3. Strong persistent discrepancy
    if row["Temporal"] > 0.70:
        return "RECALIBRATE"

    # 4. Strong context-dependent structural discrepancy
    if row["Context"] > 0.70 and row["Temporal"] > 0.20:
        return "REVISE"

    # 5. Not enough evidence
    return "ABSTAIN"


df["Predicted"] = df.apply(etmr_decision, axis=1)

df["Correct"] = df["Expected"] == df["Predicted"]

print()
print(df[
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
].to_string(index=False))

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

df.to_csv("step14b_etmr_benchmark_results.csv", index=False)

print()
print("Saved: step14b_etmr_benchmark_results.csv")
print("=" * 80)
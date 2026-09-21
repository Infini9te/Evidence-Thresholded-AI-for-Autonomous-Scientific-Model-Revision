import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 18: PROVENANCE THRESHOLD SENSITIVITY")
print("=" * 80)

df = pd.read_csv("step14_parameterized_benchmark.csv")


def etmr_decision(row, provenance_threshold):

    quality = row["Provenance"]
    persistence = row["Temporal"]
    context = row["Context"]

    # Evidence quality gate
    if quality < provenance_threshold:
        return "ABSTAIN"

    # Persistent discrepancy
    if persistence > 0.70:
        return "RECALIBRATE"

    # Structural discrepancy
    if (
        context > 0.70
        and persistence > 0.20
    ):
        return "REVISE"

    # Weak / unstructured discrepancy
    if (
        persistence < 0.20
        and context < 0.20
    ):
        return "KEEP"

    return "ABSTAIN"


results = []

thresholds = np.arange(0.10, 1.00, 0.05)

for threshold in thresholds:

    df["Predicted"] = df.apply(
        lambda row: etmr_decision(row, threshold),
        axis=1
    )

    accuracy = (
        df["Predicted"] == df["Expected"]
    ).mean()

    results.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy,
        "Correct": int(
            (df["Predicted"] == df["Expected"]).sum()
        )
    })


results_df = pd.DataFrame(results)

print()

print(results_df.to_string(index=False))

print()
print("=" * 80)

best = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print(
    f"Best threshold in this benchmark: "
    f"{best['Threshold']:.2f}"
)

print(
    f"Accuracy at best threshold: "
    f"{best['Accuracy']:.3f}"
)

print(
    f"Correct decisions: "
    f"{int(best['Correct'])} / {len(df)}"
)

print("=" * 80)


results_df.to_csv(
    "step18_threshold_sensitivity.csv",
    index=False
)

print()
print("Saved: step18_threshold_sensitivity.csv")
print("=" * 80)
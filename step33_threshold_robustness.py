import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 33: THRESHOLD ROBUSTNESS")
print("=" * 80)

df = pd.read_csv(
    "step32_provenance_aware_etmr_results.csv"
)

# ---------------------------------------------------------
# DECISION FUNCTION
# ---------------------------------------------------------

def decide(row, quality_threshold):

    quality = row["Mean_Reliability"]
    persistence = row["Temporal_Persistence"]
    context = row["Context_Dependence"]

    # Evidence too unreliable
    if quality < quality_threshold:
        return "ABSTAIN"

    # Strong persistent discrepancy
    if persistence >= 0.70:
        return "RECALIBRATE"

    # Strong contextual discrepancy
    if (
        context >= 0.70
        and
        persistence >= 0.20
    ):
        return "REVISE"

    # Weak / unstructured discrepancy
    if (
        persistence < 0.20
        and
        context < 0.20
    ):
        return "KEEP"

    # Ambiguous
    return "ABSTAIN"


# ---------------------------------------------------------
# TEST MANY THRESHOLDS
# ---------------------------------------------------------

thresholds = np.arange(
    0.00,
    0.31,
    0.01
)

results = []

for threshold in thresholds:

    predictions = []

    for _, row in df.iterrows():

        predictions.append(
            decide(row, threshold)
        )

    accuracy = np.mean(
        np.array(predictions)
        ==
        df["Expected"].values
    )

    results.append({
        "Quality_Threshold": threshold,
        "Accuracy": accuracy,
        "Correct": int(
            accuracy * len(df)
        )
    })


results_df = pd.DataFrame(results)


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("\nThreshold sensitivity:")
print(
    results_df.to_string(index=False)
)


# ---------------------------------------------------------
# BEST THRESHOLDS
# ---------------------------------------------------------

max_accuracy = results_df["Accuracy"].max()

best = results_df[
    results_df["Accuracy"] == max_accuracy
]

print("\n" + "=" * 80)
print("BEST THRESHOLDS")
print("=" * 80)

print(best.to_string(index=False))


# ---------------------------------------------------------
# ROBUST RANGE
# ---------------------------------------------------------

perfect = results_df[
    results_df["Accuracy"] == 1.0
]

print("\n" + "=" * 80)
print("THRESHOLDS WITH 100% ACCURACY")
print("=" * 80)

if len(perfect) > 0:

    print(
        f"Minimum threshold: "
        f"{perfect['Quality_Threshold'].min():.2f}"
    )

    print(
        f"Maximum threshold: "
        f"{perfect['Quality_Threshold'].max():.2f}"
    )

else:

    print("No threshold achieved 100%.")


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

results_df.to_csv(
    "step33_threshold_robustness.csv",
    index=False
)

print("\nSaved:")
print("step33_threshold_robustness.csv")

print("\n" + "=" * 80)
print("STEP 33 COMPLETE")
print("=" * 80)
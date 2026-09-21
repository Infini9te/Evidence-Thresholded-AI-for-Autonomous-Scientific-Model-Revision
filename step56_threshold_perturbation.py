import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 56: INDEPENDENT THRESHOLD PERTURBATION TEST")
print("=" * 80)

rng = np.random.default_rng(20260911)

cases = []

# ============================================================
# 1. RECALIBRATE CASES
# ============================================================

for i in range(40):

    temporal = rng.uniform(0.55, 0.90)
    lagged = rng.uniform(0.00, 0.30)
    interaction = rng.uniform(0.00, 0.30)
    context = rng.uniform(0.05, 0.40)

    cases.append({
        "World": f"RECALIBRATE_{i+1:02d}",
        "Expected": "RECALIBRATE",
        "Temporal": temporal,
        "Context": context,
        "Lagged_Rainfall": lagged,
        "Interaction": interaction,
        "Q_evidence": rng.uniform(0.90, 1.00)
    })


# ============================================================
# 2. REVISE CASES
# ============================================================

for i in range(40):

    mechanism = i % 3

    if mechanism == 0:
        # Strong contemporaneous structural effect
        context = rng.uniform(0.55, 0.95)
        lagged = rng.uniform(0.00, 0.40)
        interaction = rng.uniform(0.00, 0.40)

    elif mechanism == 1:
        # Strong delayed effect
        context = rng.uniform(0.05, 0.40)
        lagged = rng.uniform(0.55, 0.95)
        interaction = rng.uniform(0.00, 0.30)

    else:
        # Strong interaction
        context = rng.uniform(0.40, 0.80)
        lagged = rng.uniform(0.05, 0.40)
        interaction = rng.uniform(0.55, 0.90)

    cases.append({
        "World": f"REVISE_{i+1:02d}",
        "Expected": "REVISE",
        "Temporal": rng.uniform(0.20, 0.80),
        "Context": context,
        "Lagged_Rainfall": lagged,
        "Interaction": interaction,
        "Q_evidence": rng.uniform(0.90, 1.00)
    })


# ============================================================
# 3. KEEP CASES
# ============================================================

for i in range(20):

    cases.append({
        "World": f"KEEP_{i+1:02d}",
        "Expected": "KEEP",
        "Temporal": rng.uniform(0.00, 0.30),
        "Context": rng.uniform(0.00, 0.20),
        "Lagged_Rainfall": rng.uniform(0.00, 0.20),
        "Interaction": rng.uniform(0.00, 0.20),
        "Q_evidence": rng.uniform(0.90, 1.00)
    })


# ============================================================
# 4. ABSTAIN CASES
# ============================================================

for i in range(20):

    cases.append({
        "World": f"ABSTAIN_{i+1:02d}",
        "Expected": "ABSTAIN",
        "Temporal": rng.uniform(0.30, 0.80),
        "Context": rng.uniform(0.50, 0.95),
        "Lagged_Rainfall": rng.uniform(0.00, 0.70),
        "Interaction": rng.uniform(0.00, 0.70),
        "Q_evidence": rng.uniform(0.05, 0.75)
    })


# ============================================================
# Create dataframe
# ============================================================

df = pd.DataFrame(cases)

print("\nTotal cases:", len(df))

print("\nClass distribution:")
print(df["Expected"].value_counts())


# ============================================================
# Evidence calculations
# ============================================================

df["Parameter_Gain"] = (
    df["Temporal"]
    * (1 - df["Lagged_Rainfall"])
    * (1 - df["Interaction"])
)

df["Structural_Gain"] = (
    df["Context"]
    * (1 - df["Parameter_Gain"])
)

df["Structural_Gain"] = np.maximum(
    df["Structural_Gain"],
    df["Lagged_Rainfall"]
)

df["Structural_Gain"] = np.maximum(
    df["Structural_Gain"],
    df["Interaction"]
)


# ============================================================
# Test several candidate thresholds
# ============================================================

thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

results = []

for threshold in thresholds:

    predictions = []

    for _, row in df.iterrows():

        q = row["Q_evidence"]
        parameter_gain = row["Parameter_Gain"]
        structural_gain = row["Structural_Gain"]

        if q < 0.80:
            prediction = "ABSTAIN"

        elif (
            parameter_gain >= threshold
            and structural_gain < 0.30
        ):
            prediction = "RECALIBRATE"

        elif structural_gain >= 0.30:
            prediction = "REVISE"

        elif (
            parameter_gain < 0.30
            and structural_gain < 0.30
        ):
            prediction = "KEEP"

        else:
            prediction = "ABSTAIN"

        predictions.append(prediction)

    temp = df.copy()
    temp["Predicted"] = predictions

    accuracy = (
        temp["Predicted"] == temp["Expected"]
    ).mean()

    false_revision = (
        (
            (temp["Predicted"] == "REVISE")
            & (temp["Expected"] != "REVISE")
        ).sum()
    )

    missed_revision = (
        (
            (temp["Expected"] == "REVISE")
            & (temp["Predicted"] != "REVISE")
        ).sum()
    )

    correct_recalibration = (
        (
            (temp["Expected"] == "RECALIBRATE")
            & (temp["Predicted"] == "RECALIBRATE")
        ).sum()
    )

    missed_recalibration = (
        (
            (temp["Expected"] == "RECALIBRATE")
            & (temp["Predicted"] != "RECALIBRATE")
        ).sum()
    )

    results.append({
        "Parameter_Threshold": threshold,
        "Accuracy": accuracy,
        "Correct_RECALIBRATE": correct_recalibration,
        "Missed_RECALIBRATE": missed_recalibration,
        "False_REVISE": false_revision,
        "Missed_REVISE": missed_revision
    })


# ============================================================
# Display
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "-" * 80)
print("INDEPENDENT THRESHOLD PERTURBATION RESULTS")
print("-" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Parameter_Threshold": "{:.2f}".format,
            "Accuracy": "{:.4f}".format
        }
    )
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    "step56_threshold_perturbation_benchmark.csv",
    index=False
)

results_df.to_csv(
    "step56_threshold_perturbation_results.csv",
    index=False
)

print("\nSaved:")
print("step56_threshold_perturbation_benchmark.csv")
print("step56_threshold_perturbation_results.csv")

print("\n" + "=" * 80)
print("STEP 56 COMPLETE")
print("=" * 80)
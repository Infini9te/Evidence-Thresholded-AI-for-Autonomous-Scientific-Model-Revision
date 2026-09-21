import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 50: LARGE INDEPENDENT CONTROLLED BENCHMARK")
print("=" * 80)

rng = np.random.default_rng(20260910)

cases = []

# ---------------------------------------------------------
# 1. KEEP — OBSERVATION NOISE
# ---------------------------------------------------------

noise_levels = np.linspace(0.01, 0.30, 25)

for i, severity in enumerate(noise_levels):

    temporal = rng.uniform(0.00, 0.08)
    context = rng.uniform(0.00, 0.10)
    lagged = rng.uniform(0.00, 0.08)
    interaction = rng.uniform(0.00, 0.08)

    cases.append({
        "World": f"KEEP_{i+1:02d}",
        "Expected": "KEEP",
        "MAE": severity * 0.020,
        "Temporal": temporal,
        "Context": context,
        "Lagged_Rainfall": lagged,
        "Interaction": interaction,
        "Q_evidence": 1.0
    })


# ---------------------------------------------------------
# 2. RECALIBRATE — PARAMETER ERROR
# ---------------------------------------------------------

parameter_levels = np.linspace(0.01, 0.30, 25)

for i, severity in enumerate(parameter_levels):

    cases.append({
        "World": f"RECALIBRATE_{i+1:02d}",
        "Expected": "RECALIBRATE",
        "MAE": severity * 0.020,
        "Temporal": rng.uniform(0.82, 0.98),
        "Context": rng.uniform(0.75, 1.00),
        "Lagged_Rainfall": rng.uniform(0.00, 0.10),
        "Interaction": rng.uniform(0.00, 0.10),
        "Q_evidence": 1.0
    })


# ---------------------------------------------------------
# 3. REVISE — MULTIPLE STRUCTURAL MECHANISMS
# ---------------------------------------------------------

mechanisms = [
    "Rainfall_Nonlinearity",
    "Temperature_Nonlinearity",
    "Rainfall_Temperature_Interaction",
    "Rainfall_Threshold",
    "Delayed_Rainfall"
]

for i in range(25):

    mechanism = mechanisms[i % len(mechanisms)]

    if mechanism == "Delayed_Rainfall":
        context = rng.uniform(0.00, 0.25)
        lagged = rng.uniform(0.72, 0.98)
        interaction = rng.uniform(0.00, 0.20)

    elif mechanism == "Rainfall_Temperature_Interaction":
        context = rng.uniform(0.72, 0.98)
        lagged = rng.uniform(0.10, 0.45)
        interaction = rng.uniform(0.62, 0.90)

    else:
        context = rng.uniform(0.72, 1.00)
        lagged = rng.uniform(0.05, 0.40)
        interaction = rng.uniform(0.05, 0.45)

    cases.append({
        "World": f"REVISE_{i+1:02d}_{mechanism}",
        "Expected": "REVISE",
        "MAE": rng.uniform(0.003, 0.060),
        "Temporal": rng.uniform(0.20, 0.98),
        "Context": context,
        "Lagged_Rainfall": lagged,
        "Interaction": interaction,
        "Q_evidence": 1.0
    })


# ---------------------------------------------------------
# 4. ABSTAIN — INSUFFICIENT EVIDENCE
# ---------------------------------------------------------

quality_levels = np.linspace(0.05, 0.75, 25)

for i, quality in enumerate(quality_levels):

    cases.append({
        "World": f"ABSTAIN_{i+1:02d}",
        "Expected": "ABSTAIN",
        "MAE": rng.uniform(0.010, 0.040),
        "Temporal": rng.uniform(0.25, 0.60),
        "Context": rng.uniform(0.75, 0.98),
        "Lagged_Rainfall": rng.uniform(0.00, 0.30),
        "Interaction": rng.uniform(0.00, 0.30),
        "Q_evidence": quality
    })


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame(cases)

print("\nTotal cases:", len(df))

print("\nClass distribution:")
print(df["Expected"].value_counts())

print("\nQuality range:")
print(
    df.groupby("Expected")["Q_evidence"]
    .agg(["min", "max", "mean"])
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv(
    "step50_large_independent_benchmark.csv",
    index=False
)

print("\nSaved:")
print("step50_large_independent_benchmark.csv")

print("\n" + "=" * 80)
print("STEP 50 COMPLETE")
print("=" * 80)
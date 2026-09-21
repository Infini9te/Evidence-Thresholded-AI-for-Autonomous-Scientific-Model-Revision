import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 19: UNSEEN GENERALIZATION BENCHMARK")
print("=" * 80)

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

true_sm = df["sm_surface"].copy()
rainfall = df["rainfall"].fillna(0)

# IMPORTANT:
# Different random seed from Step 14.
rng = np.random.default_rng(84721)

worlds = []


# =========================================================
# WORLD A — OBSERVATION NOISE
# =========================================================

noise_levels = [
    0.03,
    0.07,
    0.12,
    0.17,
    0.22
]

for level in noise_levels:

    noise_std = true_sm.std() * level

    noise = rng.normal(
        0,
        noise_std,
        len(df)
    )

    worlds.append({
        "World": f"A_UnseenNoise_{level:.2f}",
        "Expected": "KEEP",
        "Severity": level,
        "Residual": noise,
        "Provenance": 1.0
    })


# =========================================================
# WORLD B — PARAMETER ERROR
# =========================================================

parameter_errors = [
    0.03,
    0.07,
    0.12,
    0.17,
    0.22
]

for error in parameter_errors:

    parameter_scale = 1.0 - error

    prediction = (
        true_sm.mean()
        + parameter_scale
        * (true_sm - true_sm.mean())
    )

    residual = true_sm - prediction

    worlds.append({
        "World": f"B_UnseenParameter_{error:.2f}",
        "Expected": "RECALIBRATE",
        "Severity": error,
        "Residual": residual,
        "Provenance": 1.0
    })


# =========================================================
# WORLD C — STRUCTURAL ERROR
# =========================================================

structural_levels = [
    0.003,
    0.007,
    0.012,
    0.018,
    0.030
]

for level in structural_levels:

    mechanism = (
        level
        * np.sqrt(np.maximum(rainfall, 0))
    )

    residual = mechanism

    worlds.append({
        "World": f"C_UnseenStructural_{level:.3f}",
        "Expected": "REVISE",
        "Severity": level,
        "Residual": residual,
        "Provenance": 1.0
    })


# =========================================================
# WORLD D — INSUFFICIENT EVIDENCE
# =========================================================

provenance_levels = [
    0.75,
    0.65,
    0.55,
    0.45,
    0.25
]

structural_effect = (
    0.018
    * np.sqrt(np.maximum(rainfall, 0))
)

for quality in provenance_levels:

    worlds.append({
        "World": f"D_UnseenInsufficient_{quality:.2f}",
        "Expected": "ABSTAIN",
        "Severity": quality,
        "Residual": structural_effect,
        "Provenance": quality
    })


# =========================================================
# EXTRACT EVIDENCE
# =========================================================

results = []

for world in worlds:

    residual = pd.Series(world["Residual"])

    mae = residual.abs().mean()

    temporal = abs(
        residual.autocorr(lag=1)
    )

    if np.isnan(temporal):
        temporal = 0.0

    correlations = []

    for column in [
        "rainfall",
        "temperature",
        "solar_radiation",
        "sm_surface_wetness",
        "rainfall_7d_sum",
        "ndvi"
    ]:

        corr = abs(
            residual.corr(df[column])
        )

        if not np.isnan(corr):
            correlations.append(corr)

    context = (
        max(correlations)
        if correlations
        else 0.0
    )

    results.append({
        "World": world["World"],
        "Expected": world["Expected"],
        "Severity": world["Severity"],
        "MAE": mae,
        "Temporal": temporal,
        "Context": context,
        "Provenance": world["Provenance"]
    })


benchmark = pd.DataFrame(results)

print()

print(
    benchmark.to_string(index=False)
)

print()
print("=" * 80)

print(
    f"Total unseen cases: {len(benchmark)}"
)

print("=" * 80)

benchmark.to_csv(
    "step19_unseen_benchmark.csv",
    index=False
)

print()
print("Saved: step19_unseen_benchmark.csv")
print("=" * 80)
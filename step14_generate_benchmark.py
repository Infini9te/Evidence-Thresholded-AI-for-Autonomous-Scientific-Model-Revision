import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 14A: PARAMETERIZED CONTROLLED BENCHMARK")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

true_sm = df["sm_surface"].copy()
rainfall = df["rainfall"].fillna(0)


# ============================================================
# BASE PARAMETERS
# ============================================================

rng = np.random.default_rng(2026)

worlds = []


# ============================================================
# WORLD A — OBSERVATION NOISE
# ============================================================

noise_levels = [
    0.02,
    0.05,
    0.10,
    0.15,
    0.20
]

for level in noise_levels:

    noise_std = true_sm.std() * level

    noise = rng.normal(
        0,
        noise_std,
        len(df)
    )

    residual = noise

    worlds.append({

        "World": f"A_Noise_{level:.2f}",

        "Type": "KEEP",

        "Severity": level,

        "Residual": residual,

        "Provenance": 1.0
    })


# ============================================================
# WORLD B — PARAMETER ERROR
# ============================================================

parameter_errors = [
    0.02,
    0.05,
    0.10,
    0.15,
    0.20
]

for error in parameter_errors:

    parameter_scale = 1.0 - error

    prediction = (
        true_sm.mean()
        +
        parameter_scale
        *
        (true_sm - true_sm.mean())
    )

    residual = (
        true_sm - prediction
    )

    worlds.append({

        "World": f"B_Parameter_{error:.2f}",

        "Type": "RECALIBRATE",

        "Severity": error,

        "Residual": residual,

        "Provenance": 1.0
    })


# ============================================================
# WORLD C — STRUCTURAL ERROR
# ============================================================

structural_levels = [
    0.002,
    0.005,
    0.010,
    0.015,
    0.025
]

for level in structural_levels:

    mechanism = (
        level
        *
        np.sqrt(
            np.maximum(rainfall, 0)
        )
    )

    residual = mechanism

    worlds.append({

        "World": f"C_Structural_{level:.3f}",

        "Type": "REVISE",

        "Severity": level,

        "Residual": residual,

        "Provenance": 1.0
    })


# ============================================================
# WORLD D — STRUCTURAL-LOOKING ERROR + BAD EVIDENCE
# ============================================================

provenance_levels = [
    0.80,
    0.60,
    0.40,
    0.20,
    0.10
]

structural_effect = (
    0.015
    *
    np.sqrt(
        np.maximum(rainfall, 0)
    )
)

for quality in provenance_levels:

    worlds.append({

        "World": f"D_Insufficient_{quality:.2f}",

        "Type": "ABSTAIN",

        "Severity": quality,

        "Residual": structural_effect,

        "Provenance": quality
    })


# ============================================================
# CALCULATE EVIDENCE
# ============================================================

results = []


for world in worlds:

    residual = pd.Series(
        world["Residual"]
    )

    mae = residual.abs().mean()

    temporal = abs(
        residual.autocorr(
            lag=1
        )
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
            residual.corr(
                df[column]
            )
        )

        if not np.isnan(corr):

            correlations.append(
                corr
            )


    context = (
        max(correlations)
        if correlations
        else 0.0
    )


    results.append({

        "World": world["World"],

        "Expected": world["Type"],

        "Severity": world["Severity"],

        "MAE": mae,

        "Temporal": temporal,

        "Context": context,

        "Provenance": world["Provenance"]
    })


# ============================================================
# SAVE
# ============================================================

benchmark = pd.DataFrame(
    results
)

benchmark.to_csv(
    "step14_parameterized_benchmark.csv",
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print()

print(
    benchmark.to_string(
        index=False
    )
)

print()

print(
    f"Total benchmark cases: {len(benchmark)}"
)

print()

print("=" * 80)

print(
    "Saved: step14_parameterized_benchmark.csv"
)

print("=" * 80)
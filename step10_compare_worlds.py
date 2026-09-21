import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 10: CONTROLLED WORLD COMPARISON")
print("=" * 80)


# ============================================================
# FUNCTION: CALCULATE EVIDENCE
# ============================================================

def calculate_evidence(df, residual_column):

    residual = df[residual_column]

    # --------------------------------------------------------
    # Error magnitude
    # --------------------------------------------------------

    mae = residual.abs().mean()

    # --------------------------------------------------------
    # Temporal persistence
    # --------------------------------------------------------

    temporal = abs(residual.autocorr(lag=1))

    if np.isnan(temporal):
        temporal = 0.0

    # --------------------------------------------------------
    # Context dependence
    # --------------------------------------------------------

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

    context = max(correlations) if correlations else 0.0

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    observed_fraction = df["is_observed"].mean()

    pixel_quality = df["valid_pixel_fraction"].mean()

    image_quality = min(
        df["n_images"].mean() / 10,
        1.0
    )

    provenance = (
        0.5 * observed_fraction
        +
        0.3 * pixel_quality
        +
        0.2 * image_quality
    )

    return {
        "MAE": mae,
        "Temporal": temporal,
        "Context": context,
        "Provenance": provenance
    }


# ============================================================
# WORLD A
# ============================================================

world_a = pd.read_csv(
    "step5_world_A_noise.csv"
)

# World A uses the difference between noisy observation
# and the original soil moisture.

world_a["residual"] = (
    world_a["sm_surface_noisy"]
    -
    world_a["sm_surface"]
)

a = calculate_evidence(
    world_a,
    "residual"
)


# ============================================================
# WORLD B
# ============================================================

world_b = pd.read_csv(
    "step6_world_B_parameter_error.csv"
)

b = calculate_evidence(
    world_b,
    "residual_parameter_error"
)


# ============================================================
# WORLD C
# ============================================================

world_c = pd.read_csv(
    "step7_world_C_structural_error.csv"
)

c = calculate_evidence(
    world_c,
    "residual_structural_error"
)


# ============================================================
# WORLD D
# ============================================================

world_d = pd.read_csv(
    "step8_world_D_insufficient_evidence.csv"
)

d = calculate_evidence(
    world_d,
    "residual_uncertain"
)


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

results = pd.DataFrame({

    "World": [
        "A_Noise",
        "B_Parameter",
        "C_Structural",
        "D_Insufficient"
    ],

    "MAE": [
        a["MAE"],
        b["MAE"],
        c["MAE"],
        d["MAE"]
    ],

    "Temporal": [
        a["Temporal"],
        b["Temporal"],
        c["Temporal"],
        d["Temporal"]
    ],

    "Context": [
        a["Context"],
        b["Context"],
        c["Context"],
        d["Context"]
    ],

    "Provenance": [
        a["Provenance"],
        b["Provenance"],
        c["Provenance"],
        d["Provenance"]
    ],

    "Expected_Decision": [
        "KEEP",
        "RECALIBRATE",
        "REVISE",
        "ABSTAIN"
    ]

})


# ============================================================
# DISPLAY
# ============================================================

print()
print(results.to_string(index=False))

# ============================================================
# SAVE
# ============================================================

results.to_csv(
    "step10_world_comparison.csv",
    index=False
)

print()
print("=" * 80)
print("Saved: step10_world_comparison.csv")
print("=" * 80)
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

print("=" * 80)
print("ETMR — STEP 28: CONTROLLED WORLD EVIDENCE PROFILES")
print("=" * 80)

files = {
    "A_Noise": "step5_world_A_noise.csv",
    "B_Parameter": "step6_world_B_parameter_error.csv",
    "C_Structural": "step7_world_C_structural_error.csv",
    "D_Insufficient": "step8_world_D_insufficient_evidence.csv"
}

expected = {
    "A_Noise": "KEEP",
    "B_Parameter": "RECALIBRATE",
    "C_Structural": "REVISE",
    "D_Insufficient": "ABSTAIN"
}

residual_columns = {
    "A_Noise": ("sm_surface", "sm_surface_noisy"),
    "B_Parameter": ("sm_surface", "sm_surface_parameter_error"),
    "C_Structural": ("sm_surface", "sm_surface_structural"),
    "D_Insufficient": ("sm_surface", "sm_surface_uncertain")
}

profiles = []

for world, filename in files.items():

    print("\n" + "=" * 80)
    print(world)
    print("=" * 80)

    df = pd.read_csv(filename)

    true_col, prediction_col = residual_columns[world]

    # ---------------------------------------------------------
    # RESIDUAL
    # ---------------------------------------------------------

    true_values = pd.to_numeric(
        df[true_col],
        errors="coerce"
    )

    predicted_values = pd.to_numeric(
        df[prediction_col],
        errors="coerce"
    )

    residual = (
        true_values - predicted_values
    )

    valid_residual = residual.dropna()

    # ---------------------------------------------------------
    # BASIC RESIDUAL EVIDENCE
    # ---------------------------------------------------------

    mae = np.mean(np.abs(valid_residual))
    bias = np.mean(valid_residual)
    std = np.std(valid_residual)

    temporal = abs(
        valid_residual.autocorr(lag=1)
    )

    # ---------------------------------------------------------
    # CONTEXT DEPENDENCE
    # ---------------------------------------------------------

    context_values = {}

    for variable in [
        "rainfall",
        "temperature",
        "solar_radiation"
    ]:

        if variable not in df.columns:
            continue

        x = pd.to_numeric(
            df[variable],
            errors="coerce"
        )

        valid = (
            x.notna()
            &
            residual.notna()
        )

        if valid.sum() > 2:

            corr = pearsonr(
                x.loc[valid],
                residual.loc[valid]
            )[0]

            if np.isfinite(corr):
                context_values[variable] = abs(corr)

    if context_values:

        context = max(
            context_values.values()
        )

        strongest_context = max(
            context_values,
            key=context_values.get
        )

    else:

        context = np.nan
        strongest_context = "None"

    # ---------------------------------------------------------
    # PROVENANCE
    # ---------------------------------------------------------

    provenance = 1.0

    if "evidence_quality" in df.columns:

        quality = pd.to_numeric(
            df["evidence_quality"],
            errors="coerce"
        )

        if quality.notna().any():
            provenance = quality.mean()

    elif "Controlled_Provenance" in df.columns:

        quality = pd.to_numeric(
            df["Controlled_Provenance"],
            errors="coerce"
        )

        if quality.notna().any():
            provenance = quality.mean()

    elif "valid_pixel_fraction" in df.columns:

        quality = pd.to_numeric(
            df["valid_pixel_fraction"],
            errors="coerce"
        )

        if quality.notna().any():
            provenance = quality.mean()

    # ---------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------

    print(f"MAE                  : {mae:.9f}")
    print(f"Bias                 : {bias:.9f}")
    print(f"Residual std         : {std:.9f}")
    print(f"Temporal persistence : {temporal:.6f}")
    print(f"Context dependence   : {context:.6f}")
    print(f"Strongest context    : {strongest_context}")
    print(f"Provenance quality   : {provenance:.6f}")
    print(f"Expected decision    : {expected[world]}")

    profiles.append({
        "World": world,
        "Expected": expected[world],
        "MAE": mae,
        "Bias": bias,
        "Residual_Std": std,
        "Temporal_Persistence": temporal,
        "Context_Dependence": context,
        "Strongest_Context": strongest_context,
        "Provenance": provenance
    })


# =============================================================
# CREATE FINAL PROFILE TABLE
# =============================================================

profiles_df = pd.DataFrame(profiles)

print("\n" + "=" * 80)
print("UNIFIED EVIDENCE PROFILES")
print("=" * 80)

print(
    profiles_df.to_string(index=False)
)

# -------------------------------------------------------------
# SAVE
# -------------------------------------------------------------

profiles_df.to_csv(
    "step28_controlled_evidence_profiles.csv",
    index=False
)

print("\nSaved:")
print("step28_controlled_evidence_profiles.csv")

print("\n" + "=" * 80)
print("STEP 28 COMPLETE")
print("=" * 80)
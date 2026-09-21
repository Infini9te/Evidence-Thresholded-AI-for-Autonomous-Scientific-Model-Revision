import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 30: RELATIVE PROVENANCE QUALITY")
print("=" * 80)

files = {
    "A_Noise": "step5_world_A_noise.csv",
    "B_Parameter": "step6_world_B_parameter_error.csv",
    "C_Structural": "step7_world_C_structural_error.csv",
    "D_Insufficient": "step8_world_D_insufficient_evidence.csv"
}

results = []

for world, filename in files.items():

    print("\n" + "=" * 80)
    print(world)
    print("=" * 80)

    df = pd.read_csv(filename)

    # ---------------------------------------------------------
    # OBSERVATION STATUS
    # ---------------------------------------------------------

    observed = pd.to_numeric(
        df["is_observed"],
        errors="coerce"
    ).mean()

    # ---------------------------------------------------------
    # PIXEL VALIDITY
    # ---------------------------------------------------------

    pixel = pd.to_numeric(
        df["valid_pixel_fraction"],
        errors="coerce"
    ).mean()

    # ---------------------------------------------------------
    # IMAGE SUPPORT
    # ---------------------------------------------------------

    images = pd.to_numeric(
        df["n_images"],
        errors="coerce"
    ).mean()

    # ---------------------------------------------------------
    # RELATIVE QUALITY
    #
    # Normalize each quantity relative to the maximum
    # observed across the four worlds.
    # ---------------------------------------------------------

    results.append({
        "World": world,
        "Observed": observed,
        "Pixel": pixel,
        "Images": images
    })


raw = pd.DataFrame(results)

# -------------------------------------------------------------
# NORMALIZATION
# -------------------------------------------------------------

raw["Observed_Rel"] = (
    raw["Observed"] /
    raw["Observed"].max()
)

raw["Pixel_Rel"] = (
    raw["Pixel"] /
    raw["Pixel"].max()
)

raw["Images_Rel"] = (
    raw["Images"] /
    raw["Images"].max()
)

# -------------------------------------------------------------
# QUALITY SCORE
# -------------------------------------------------------------

raw["Provenance_Quality"] = (
    0.40 * raw["Observed_Rel"]
    +
    0.40 * raw["Pixel_Rel"]
    +
    0.20 * raw["Images_Rel"]
)

# -------------------------------------------------------------
# DISPLAY
# -------------------------------------------------------------

print("\n" + "=" * 80)
print("RELATIVE PROVENANCE QUALITY")
print("=" * 80)

print(
    raw.to_string(index=False)
)

raw.to_csv(
    "step30_relative_provenance.csv",
    index=False
)

print("\nSaved:")
print("step30_relative_provenance.csv")

print("\n" + "=" * 80)
print("STEP 30 COMPLETE")
print("=" * 80)
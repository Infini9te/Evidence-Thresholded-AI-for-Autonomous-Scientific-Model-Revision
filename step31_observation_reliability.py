import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 31: OBSERVATION RELIABILITY")
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
    ).fillna(0)

    interpolated = pd.to_numeric(
        df["is_interpolated"],
        errors="coerce"
    ).fillna(0)

    # ---------------------------------------------------------
    # PIXEL QUALITY
    # ---------------------------------------------------------

    pixel = pd.to_numeric(
        df["valid_pixel_fraction"],
        errors="coerce"
    ).fillna(0)

    # ---------------------------------------------------------
    # IMAGE SUPPORT
    # ---------------------------------------------------------

    images = pd.to_numeric(
        df["n_images"],
        errors="coerce"
    ).fillna(0)

    image_support = np.clip(
        images / 10.0,
        0,
        1
    )

    # ---------------------------------------------------------
    # FILL METHOD
    # ---------------------------------------------------------

    fill = df["fill_method"].astype(str).str.lower()

    fill_score = np.where(
        fill == "observed",
        1.0,
        np.where(
            fill == "interpolated",
            0.5,
            np.where(
                fill == "climatology",
                0.1,
                0.0
            )
        )
    )

    # ---------------------------------------------------------
    # TIME-POINT RELIABILITY
    #
    # The observation/fill method is the primary signal.
    # Pixel quality and image support modify confidence.
    # ---------------------------------------------------------

    reliability = (
        0.50 * fill_score
        +
        0.30 * pixel
        +
        0.20 * image_support
    )

    # Interpolated/climatological observations cannot receive
    # full reliability even if pixel/image fields look good.

    reliability = np.where(
        observed == 1,
        reliability,
        reliability * 0.5
    )

    reliability = np.clip(
        reliability,
        0,
        1
    )

    mean_reliability = reliability.mean()

    # ---------------------------------------------------------
    # HIGH-QUALITY EVIDENCE FRACTION
    # ---------------------------------------------------------

    high_quality_fraction = (
        reliability >= 0.70
    ).mean()

    low_quality_fraction = (
        reliability < 0.30
    ).mean()

    # ---------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------

    print(
        f"Mean reliability       : "
        f"{mean_reliability:.6f}"
    )

    print(
        f"High-quality fraction  : "
        f"{high_quality_fraction:.6f}"
    )

    print(
        f"Low-quality fraction   : "
        f"{low_quality_fraction:.6f}"
    )

    results.append({
        "World": world,
        "Mean_Reliability": mean_reliability,
        "High_Quality_Fraction": high_quality_fraction,
        "Low_Quality_Fraction": low_quality_fraction
    })


# =============================================================
# FINAL RESULTS
# =============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("OBSERVATION RELIABILITY RESULTS")
print("=" * 80)

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    "step31_observation_reliability.csv",
    index=False
)

print("\nSaved:")
print("step31_observation_reliability.csv")

print("\n" + "=" * 80)
print("STEP 31 COMPLETE")
print("=" * 80)
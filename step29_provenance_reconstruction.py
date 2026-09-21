import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 29: PROVENANCE RECONSTRUCTION")
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
    # OBSERVATION QUALITY
    # ---------------------------------------------------------

    observed_fraction = pd.to_numeric(
        df["is_observed"],
        errors="coerce"
    ).mean()

    # ---------------------------------------------------------
    # SPATIAL QUALITY
    # ---------------------------------------------------------

    pixel_fraction = pd.to_numeric(
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

    image_score = min(images / 10.0, 1.0)

    # ---------------------------------------------------------
    # PROVENANCE SCORE
    #
    # Observable evidence quality:
    #
    # 50% observation status
    # 30% spatial validity
    # 20% image support
    # ---------------------------------------------------------

    provenance = (
        0.50 * observed_fraction
        +
        0.30 * pixel_fraction
        +
        0.20 * image_score
    )

    print(f"Observed fraction : {observed_fraction:.6f}")
    print(f"Pixel validity    : {pixel_fraction:.6f}")
    print(f"Mean images       : {images:.6f}")
    print(f"Image score       : {image_score:.6f}")
    print(f"PROVENANCE SCORE  : {provenance:.6f}")

    results.append({
        "World": world,
        "Observed_Fraction": observed_fraction,
        "Pixel_Validity": pixel_fraction,
        "Mean_Images": images,
        "Image_Score": image_score,
        "Provenance": provenance
    })


# =============================================================
# FINAL TABLE
# =============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("PROVENANCE RECONSTRUCTION RESULTS")
print("=" * 80)

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    "step29_provenance_reconstruction.csv",
    index=False
)

print("\nSaved:")
print("step29_provenance_reconstruction.csv")

print("\n" + "=" * 80)
print("STEP 29 COMPLETE")
print("=" * 80)
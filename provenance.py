import pandas as pd

# Load dataset
df = pd.read_csv("final_dataset.csv")

print("=" * 60)
print("EVIDENCE PROVENANCE AUDIT")
print("=" * 60)

# --------------------------------------------------
# 1. NDVI data provenance
# --------------------------------------------------

print("\n1. NDVI PROVENANCE")
print("-" * 40)

print(df["fill_method"].value_counts())

# --------------------------------------------------
# 2. Satellite observation quality
# --------------------------------------------------

print("\n2. VALID PIXEL FRACTION")
print("-" * 40)

print(df["valid_pixel_fraction"].describe())

# --------------------------------------------------
# 3. Number of satellite images
# --------------------------------------------------

print("\n3. NUMBER OF IMAGES")
print("-" * 40)

print(df["n_images"].describe())

# --------------------------------------------------
# 4. Observation availability
# --------------------------------------------------

print("\n4. OBSERVATION STATUS")
print("-" * 40)

print(
    df[
        ["is_observed", "is_interpolated"]
    ].value_counts()
)

# --------------------------------------------------
# 5. Compare quality by provenance
# --------------------------------------------------

print("\n5. QUALITY BY FILL METHOD")
print("-" * 40)

quality_summary = (
    df.groupby("fill_method")
    [["valid_pixel_fraction", "n_images"]]
    .agg(["mean", "median", "min", "max"])
)

print(quality_summary)

# --------------------------------------------------
# 6. NDVI statistics by provenance
# --------------------------------------------------

print("\n6. NDVI BY PROVENANCE")
print("-" * 40)

ndvi_summary = (
    df.groupby("fill_method")["ndvi"]
    .agg(["count", "mean", "std", "min", "max"])
)

print(ndvi_summary)

print("\n" + "=" * 60)
print("STEP 2 AUDIT COMPLETE")
print("=" * 60)
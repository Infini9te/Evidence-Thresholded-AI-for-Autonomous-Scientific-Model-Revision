import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 21: DATA LEAKAGE AUDIT")
print("=" * 80)

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# =========================================================
# 1. Check chronological ordering
# =========================================================

print()
print("1. DATE ORDER")
print("-" * 80)

is_sorted = df["date"].is_monotonic_increasing

print("Chronologically sorted:", is_sorted)

print("First date:", df["date"].min())
print("Last date :", df["date"].max())


# =========================================================
# 2. Check duplicate dates
# =========================================================

print()
print("2. DUPLICATE DATES")
print("-" * 80)

duplicate_dates = df["date"].duplicated().sum()

print("Duplicate date rows:", duplicate_dates)


# =========================================================
# 3. Check lag correctness
# =========================================================

print()
print("3. LAG FEATURE AUDIT")
print("-" * 80)

# Expected lag-1 values
expected_surface_lag1 = df["sm_surface"].shift(1)
expected_rootzone_lag1 = df["sm_rootzone"].shift(1)
expected_ndvi_lag1 = df["ndvi"].shift(1)

surface_difference = (
    df["sm_surface_lag1"] -
    expected_surface_lag1
).abs()

rootzone_difference = (
    df["sm_rootzone_lag1"] -
    expected_rootzone_lag1
).abs()

ndvi_difference = (
    df["ndvi_lag1"] -
    expected_ndvi_lag1
).abs()


print(
    "Surface lag mismatches:",
    (surface_difference > 1e-10).sum()
)

print(
    "Rootzone lag mismatches:",
    (rootzone_difference > 1e-10).sum()
)

print(
    "NDVI lag mismatches:",
    (ndvi_difference > 1e-10).sum()
)


# =========================================================
# 4. Check future information
# =========================================================

print()
print("4. FUTURE INFORMATION CORRELATION")
print("-" * 80)

future_surface = df["sm_surface"].shift(-1)

for column in [
    "temperature",
    "rainfall",
    "solar_radiation",
    "sm_surface_wetness",
    "n_obs_3hourly",
    "ndvi",
    "rainfall_7d_sum",
    "solar_7d_mean",
    "sm_surface_lag1",
    "sm_rootzone_lag1",
    "ndvi_lag1"
]:

    corr = df[column].corr(future_surface)

    print(
        f"{column:25s}: {corr:.6f}"
    )


# =========================================================
# 5. Direct target correlations
# =========================================================

print()
print("5. FEATURE ↔ TARGET CORRELATION")
print("-" * 80)

target = df["sm_surface"]

for column in [
    "temperature",
    "rainfall",
    "solar_radiation",
    "sm_surface_wetness",
    "n_obs_3hourly",
    "ndvi",
    "rainfall_7d_sum",
    "solar_7d_mean",
    "sm_surface_lag1",
    "sm_rootzone_lag1",
    "ndvi_lag1"
]:

    corr = df[column].corr(target)

    print(
        f"{column:25s}: {corr:.6f}"
    )


# =========================================================
# 6. Check whether lag is almost identical to target
# =========================================================

print()
print("6. TARGET / LAG RELATIONSHIP")
print("-" * 80)

valid = df[
    ["sm_surface", "sm_surface_lag1"]
].dropna()

corr_lag_target = valid[
    "sm_surface"
].corr(
    valid["sm_surface_lag1"]
)

print(
    "Correlation(sm_surface, sm_surface_lag1):",
    f"{corr_lag_target:.6f}"
)


# =========================================================
# 7. Check train/test temporal separation
# =========================================================

print()
print("7. TRAIN / TEST SEPARATION")
print("-" * 80)

split_date = pd.Timestamp("2024-01-01")

train = df[df["date"] < split_date]
test = df[df["date"] >= split_date]

print("Training rows:", len(train))
print("Testing rows :", len(test))

print(
    "Training last date:",
    train["date"].max()
)

print(
    "Testing first date:",
    test["date"].min()
)


# =========================================================
# 8. Missing-value check
# =========================================================

print()
print("8. MISSING VALUES")
print("-" * 80)

print(
    df.isna().sum()
)


print()
print("=" * 80)
print("LEAKAGE AUDIT COMPLETE")
print("=" * 80)
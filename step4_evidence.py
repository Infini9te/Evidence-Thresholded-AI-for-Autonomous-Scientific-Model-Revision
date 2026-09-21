import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD STEP 3 RESULTS
# ============================================================

df = pd.read_csv("step3_predictions.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 2. RESIDUAL
# ============================================================

r = df["residual"]


# ============================================================
# 3. BASIC RESIDUAL EVIDENCE
# ============================================================

bias = r.mean()

variance = r.var()

std = r.std()

mae = r.abs().mean()


# ============================================================
# 4. TEMPORAL AUTOCORRELATION
# ============================================================

temporal_autocorrelation = r.autocorr(lag=1)


# ============================================================
# 5. ENVIRONMENTAL DEPENDENCE
# ============================================================

rainfall_corr = r.corr(
    df["rainfall"]
)

temperature_corr = r.corr(
    df["temperature"]
)

solar_corr = r.corr(
    df["solar_radiation"]
)

wetness_corr = r.corr(
    df["sm_surface_wetness"]
)

rainfall_7d_corr = r.corr(
    df["rainfall_7d_sum"]
)


# ============================================================
# 6. SOIL-MOISTURE STATE DEPENDENCE
# ============================================================

surface_moisture_corr = r.corr(
    df["sm_surface"]
)

# NOTE:
# sm_rootzone is not present in step3_predictions.csv.
# Therefore we do NOT calculate that correlation here.


# ============================================================
# 7. NDVI DEPENDENCE
# ============================================================

ndvi_corr = r.corr(
    df["ndvi"]
)


# ============================================================
# 8. PRINT EVIDENCE VECTOR
# ============================================================

print("=" * 70)
print("ETMR — RESIDUAL EVIDENCE VECTOR")
print("=" * 70)


print("\n[1] ERROR MAGNITUDE")
print("-" * 50)

print(f"MAE       : {mae:.10f}")
print(f"Bias      : {bias:.10f}")
print(f"Variance  : {variance:.10f}")
print(f"Std       : {std:.10f}")


print("\n[2] TEMPORAL STRUCTURE")
print("-" * 50)

print(
    f"Lag-1 autocorrelation : "
    f"{temporal_autocorrelation:.6f}"
)


print("\n[3] ENVIRONMENTAL DEPENDENCE")
print("-" * 50)

print(
    f"Rainfall              : "
    f"{rainfall_corr:.6f}"
)

print(
    f"Temperature           : "
    f"{temperature_corr:.6f}"
)

print(
    f"Solar radiation       : "
    f"{solar_corr:.6f}"
)

print(
    f"Surface wetness       : "
    f"{wetness_corr:.6f}"
)

print(
    f"Rainfall 7-day sum    : "
    f"{rainfall_7d_corr:.6f}"
)


print("\n[4] SOIL-MOISTURE DEPENDENCE")
print("-" * 50)

print(
    f"Surface moisture      : "
    f"{surface_moisture_corr:.6f}"
)


print("\n[5] VEGETATION DEPENDENCE")
print("-" * 50)

print(
    f"NDVI                  : "
    f"{ndvi_corr:.6f}"
)


# ============================================================
# 9. CREATE EVIDENCE VECTOR
# ============================================================

evidence = pd.DataFrame({
    "bias": [bias],
    "variance": [variance],
    "std": [std],
    "mae": [mae],

    "temporal_autocorrelation": [
        temporal_autocorrelation
    ],

    "rainfall_correlation": [
        rainfall_corr
    ],

    "temperature_correlation": [
        temperature_corr
    ],

    "solar_correlation": [
        solar_corr
    ],

    "wetness_correlation": [
        wetness_corr
    ],

    "rainfall_7d_correlation": [
        rainfall_7d_corr
    ],

    "surface_moisture_correlation": [
        surface_moisture_corr
    ],

    "ndvi_correlation": [
        ndvi_corr
    ]
})


# ============================================================
# 10. SAVE
# ============================================================

evidence.to_csv(
    "step4_evidence_vector.csv",
    index=False
)


print("\n" + "=" * 70)
print("Evidence vector saved:")
print("step4_evidence_vector.csv")
print("=" * 70)
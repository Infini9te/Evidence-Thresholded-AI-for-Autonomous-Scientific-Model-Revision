import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

print("=" * 80)
print("ETMR — STEP 67: COUNTERFACTUAL ENGINE SANITY TEST")
print("=" * 80)

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = pd.read_csv("final_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

target = "sm_surface"

features = [
    "temperature",
    "rainfall",
    "solar_radiation",
    "n_obs_3hourly",
    "ndvi",
    "rainfall_7d_sum",
    "solar_7d_mean",
    "sm_rootzone_lag1",
    "ndvi_lag1"
]

train = df[df["date"] < "2024-01-01"].copy()

test = df[
    (df["date"] >= "2024-01-01") &
    (df["date"] < "2025-01-01")
].copy()

X_train = train[features]
y_train = train[target].values

X_test = test[features]

# ---------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

train_pred = model.predict(X_train)

train_mean = y_train.mean()

numerator = np.sum(
    (train_pred - train_mean)
    * (y_train - train_mean)
)

denominator = np.sum(
    (train_pred - train_mean) ** 2
)

alpha = numerator / denominator

print(f"\nEstimated baseline alpha = {alpha:.8f}")

# ---------------------------------------------------------------------
# Test variables
# ---------------------------------------------------------------------

y_real = test[target].values

rain = np.sqrt(
    np.maximum(
        test["rainfall"].values,
        0
    )
)

rng = np.random.default_rng(20260911)

# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def rmse(y, prediction):

    return np.sqrt(
        np.mean(
            (y - prediction) ** 2
        )
    )


# =====================================================================
# TEST 1 — PURE PARAMETER ERROR
# =====================================================================

print("\n" + "=" * 80)
print("TEST 1 — PURE PARAMETER ERROR")
print("=" * 80)

parameter_scale = 0.92

y_parameter = (
    y_real *
    parameter_scale
)

# Parameter counterfactual
parameter_prediction = (
    train_mean
    + alpha *
    (
        pred -
        train_mean
    )
)

baseline_error = rmse(
    y_parameter,
    pred
)

parameter_error = rmse(
    y_parameter,
    parameter_prediction
)

parameter_gain = (
    baseline_error -
    parameter_error
)

# Structural counterfactual
structural_prediction = (
    pred +
    0.015 * rain
)

structural_error = rmse(
    y_parameter,
    structural_prediction
)

structural_gain = (
    baseline_error -
    structural_error
)

print(f"True parameter scale = {parameter_scale:.4f}")

print(f"Baseline RMSE = {baseline_error:.8f}")

print(
    f"Parameter-counterfactual RMSE = "
    f"{parameter_error:.8f}"
)

print(
    f"Parameter gain = "
    f"{parameter_gain:.8f}"
)

print(
    f"Structural-counterfactual RMSE = "
    f"{structural_error:.8f}"
)

print(
    f"Structural gain = "
    f"{structural_gain:.8f}"
)

if parameter_gain > structural_gain:

    print("\nRESULT: PARAMETER COUNTERFACTUAL WINS")

else:

    print("\nRESULT: STRUCTURAL COUNTERFACTUAL WINS")


# =====================================================================
# TEST 2 — PURE STRUCTURAL ERROR
# =====================================================================

print("\n" + "=" * 80)
print("TEST 2 — PURE STRUCTURAL ERROR")
print("=" * 80)

structural_strength = 1.0

y_structural = (
    y_real
    + 0.015 *
    structural_strength *
    rain
)

# Parameter counterfactual
parameter_prediction = (
    train_mean
    + alpha *
    (
        pred -
        train_mean
    )
)

baseline_error = rmse(
    y_structural,
    pred
)

parameter_error = rmse(
    y_structural,
    parameter_prediction
)

parameter_gain = (
    baseline_error -
    parameter_error
)

# Structural counterfactual
structural_prediction = (
    pred +
    0.015 *
    rain
)

structural_error = rmse(
    y_structural,
    structural_prediction
)

structural_gain = (
    baseline_error -
    structural_error
)

print(
    f"Baseline RMSE = "
    f"{baseline_error:.8f}"
)

print(
    f"Parameter-counterfactual RMSE = "
    f"{parameter_error:.8f}"
)

print(
    f"Parameter gain = "
    f"{parameter_gain:.8f}"
)

print(
    f"Structural-counterfactual RMSE = "
    f"{structural_error:.8f}"
)

print(
    f"Structural gain = "
    f"{structural_gain:.8f}"
)

if structural_gain > parameter_gain:

    print("\nRESULT: STRUCTURAL COUNTERFACTUAL WINS")

else:

    print("\nRESULT: PARAMETER COUNTERFACTUAL WINS")


# =====================================================================
# TEST 3 — NO MODEL ERROR
# =====================================================================

print("\n" + "=" * 80)
print("TEST 3 — NO MODEL ERROR")
print("=" * 80)

y_keep = (
    y_real
    + rng.normal(
        0,
        0.0005,
        size=len(y_real)
    )
)

baseline_error = rmse(
    y_keep,
    pred
)

parameter_error = rmse(
    y_keep,
    parameter_prediction
)

structural_error = rmse(
    y_keep,
    structural_prediction
)

parameter_gain = (
    baseline_error -
    parameter_error
)

structural_gain = (
    baseline_error -
    structural_error
)

print(
    f"Baseline RMSE = "
    f"{baseline_error:.8f}"
)

print(
    f"Parameter gain = "
    f"{parameter_gain:.8f}"
)

print(
    f"Structural gain = "
    f"{structural_gain:.8f}"
)

print("\nEXPECTED: BOTH COUNTERFACTIVES SHOULD BE WEAK")


# =====================================================================
# SUMMARY
# =====================================================================

print("\n" + "=" * 80)
print("STEP 67 SUMMARY")
print("=" * 80)

print("""
The counterfactual engine should satisfy:

1. Pure parameter error:
   Parameter gain > structural gain

2. Pure structural error:
   Structural gain > parameter gain

3. No model error:
   Both gains should be small

If these conditions fail, the counterfactual
definitions must be corrected before further
ETMR evaluation.
""")

print("=" * 80)
print("STEP 67 COMPLETE")
print("=" * 80)
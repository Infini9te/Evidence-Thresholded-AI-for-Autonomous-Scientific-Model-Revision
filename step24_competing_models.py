import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 80)
print("ETMR — STEP 24: COMPETING SCIENTIFIC MODELS")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

target = "sm_surface"

# ---------------------------------------------------------
# 2. TEMPORAL SPLIT
# ---------------------------------------------------------

split_date = pd.Timestamp("2024-01-01")

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

# ---------------------------------------------------------
# 3. BASELINE SCIENTIFIC MODEL M0
# ---------------------------------------------------------

features_M0 = [
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

print("\nMODEL M0")
print("-" * 80)

print("Baseline environmental/state model")

for f in features_M0:
    print("-", f)

# ---------------------------------------------------------
# 4. TRAIN M0
# ---------------------------------------------------------

M0 = RandomForestRegressor(
    n_estimators=300,
    random_state=2026,
    n_jobs=-1
)

M0.fit(
    train[features_M0],
    train[target]
)

pred_M0 = M0.predict(test[features_M0])

residual_M0 = test[target].values - pred_M0

# ---------------------------------------------------------
# 5. MODEL M1
# ---------------------------------------------------------

# M1 contains the same scientific variables as M0,
# plus an explicit nonlinear rainfall mechanism.

train = train.copy()
test = test.copy()

train["rainfall_nonlinear"] = np.sqrt(
    np.maximum(train["rainfall"], 0)
)

test["rainfall_nonlinear"] = np.sqrt(
    np.maximum(test["rainfall"], 0)
)

features_M1 = features_M0 + [
    "rainfall_nonlinear"
]

print("\nMODEL M1")
print("-" * 80)

print("Baseline model + nonlinear rainfall mechanism")

for f in features_M1:
    print("-", f)

# ---------------------------------------------------------
# 6. TRAIN M1
# ---------------------------------------------------------

M1 = RandomForestRegressor(
    n_estimators=300,
    random_state=2026,
    n_jobs=-1
)

M1.fit(
    train[features_M1],
    train[target]
)

pred_M1 = M1.predict(test[features_M1])

residual_M1 = test[target].values - pred_M1

# ---------------------------------------------------------
# 7. EVALUATION
# ---------------------------------------------------------

mae_M0 = mean_absolute_error(
    test[target],
    pred_M0
)

rmse_M0 = np.sqrt(
    mean_squared_error(
        test[target],
        pred_M0
    )
)

r2_M0 = r2_score(
    test[target],
    pred_M0
)

mae_M1 = mean_absolute_error(
    test[target],
    pred_M1
)

rmse_M1 = np.sqrt(
    mean_squared_error(
        test[target],
        pred_M1
    )
)

r2_M1 = r2_score(
    test[target],
    pred_M1
)

# ---------------------------------------------------------
# 8. RESULTS
# ---------------------------------------------------------

print("\nMODEL PERFORMANCE")
print("=" * 80)

print(
    f"M0  MAE  : {mae_M0:.9f}"
)

print(
    f"M0  RMSE : {rmse_M0:.9f}"
)

print(
    f"M0  R²   : {r2_M0:.9f}"
)

print()

print(
    f"M1  MAE  : {mae_M1:.9f}"
)

print(
    f"M1  RMSE : {rmse_M1:.9f}"
)

print(
    f"M1  R²   : {r2_M1:.9f}"
)

# ---------------------------------------------------------
# 9. MODEL IMPROVEMENT
# ---------------------------------------------------------

print("\nMODEL IMPROVEMENT")
print("-" * 80)

print(
    f"MAE improvement  : {mae_M0 - mae_M1:.9f}"
)

print(
    f"RMSE improvement : {rmse_M0 - rmse_M1:.9f}"
)

print(
    f"R² improvement   : {r2_M1 - r2_M0:.9f}"
)

# ---------------------------------------------------------
# 10. RESIDUAL STATISTICS
# ---------------------------------------------------------

print("\nRESIDUAL COMPARISON")
print("-" * 80)

print(
    f"M0 residual mean : {np.mean(residual_M0):.9f}"
)

print(
    f"M0 residual std  : {np.std(residual_M0):.9f}"
)

print(
    f"M1 residual mean : {np.mean(residual_M1):.9f}"
)

print(
    f"M1 residual std  : {np.std(residual_M1):.9f}"
)

# ---------------------------------------------------------
# 11. SAVE
# ---------------------------------------------------------

results = test[
    [
        "date",
        "sm_surface",
        "rainfall",
        "temperature",
        "solar_radiation"
    ]
].copy()

results["M0_prediction"] = pred_M0
results["M1_prediction"] = pred_M1

results["M0_residual"] = residual_M0
results["M1_residual"] = residual_M1

results["M1_minus_M0"] = (
    pred_M1 - pred_M0
)

results.to_csv(
    "step24_competing_models_predictions.csv",
    index=False
)

print("\nSaved:")
print("step24_competing_models_predictions.csv")

print("\n" + "=" * 80)
print("STEP 24 COMPLETE")
print("=" * 80)
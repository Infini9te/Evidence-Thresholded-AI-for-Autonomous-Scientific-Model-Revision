import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 80)
print("ETMR — STEP 58: ACTUAL PARAMETER COUNTERFACTUAL")
print("=" * 80)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

# ============================================================
# 2. FEATURES
# ============================================================

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

target = "sm_surface"

# ============================================================
# 3. CHRONOLOGICAL SPLIT
# ============================================================

train = df[df["date"] < "2024-01-01"].copy()
test = df[df["date"] >= "2024-01-01"].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("\nTrain:", len(train))
print("Test :", len(test))


# ============================================================
# 4. BASE MODEL M0
# ============================================================

model_m0 = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model_m0.fit(X_train, y_train)

pred_m0 = model_m0.predict(X_test)

residual_m0 = y_test.values - pred_m0


# ============================================================
# 5. PARAMETER COUNTERFACTUAL
#
# We allow a global multiplicative correction:
#
# y_corrected = mean + alpha*(prediction - mean)
#
# alpha is estimated ONLY from training data.
# ============================================================

train_pred = model_m0.predict(X_train)

train_mean = y_train.mean()

numerator = np.sum(
    (train_pred - train_mean)
    * (y_train.values - train_mean)
)

denominator = np.sum(
    (train_pred - train_mean) ** 2
)

alpha = numerator / denominator

print("\nEstimated parameter correction alpha:")
print(f"{alpha:.6f}")


# ============================================================
# 6. APPLY PARAMETER CORRECTION
# ============================================================

pred_recalibrated = (
    train_mean
    + alpha * (pred_m0 - train_mean)
)

residual_recalibrated = (
    y_test.values - pred_recalibrated
)


# ============================================================
# 7. METRICS
# ============================================================

mae_m0 = mean_absolute_error(
    y_test,
    pred_m0
)

rmse_m0 = np.sqrt(
    mean_squared_error(
        y_test,
        pred_m0
    )
)

r2_m0 = r2_score(
    y_test,
    pred_m0
)


mae_recal = mean_absolute_error(
    y_test,
    pred_recalibrated
)

rmse_recal = np.sqrt(
    mean_squared_error(
        y_test,
        pred_recalibrated
    )
)

r2_recal = r2_score(
    y_test,
    pred_recalibrated
)


# ============================================================
# 8. IMPROVEMENT
# ============================================================

mae_improvement = mae_m0 - mae_recal
rmse_improvement = rmse_m0 - rmse_recal


print("\n" + "-" * 80)
print("MODEL COMPARISON")
print("-" * 80)

print("\nM0:")
print(f"MAE  = {mae_m0:.8f}")
print(f"RMSE = {rmse_m0:.8f}")
print(f"R²   = {r2_m0:.8f}")

print("\nM0 + parameter recalibration:")
print(f"MAE  = {mae_recal:.8f}")
print(f"RMSE = {rmse_recal:.8f}")
print(f"R²   = {r2_recal:.8f}")

print("\nImprovement:")
print(f"MAE improvement  = {mae_improvement:.8f}")
print(f"RMSE improvement = {rmse_improvement:.8f}")


# ============================================================
# 9. RESIDUAL STRUCTURE AFTER RECALIBRATION
# ============================================================

rainfall = test["rainfall"].values
temperature = test["temperature"].values
solar = test["solar_radiation"].values

corr_rain = np.corrcoef(
    residual_recalibrated,
    rainfall
)[0, 1]

corr_temp = np.corrcoef(
    residual_recalibrated,
    temperature
)[0, 1]

corr_solar = np.corrcoef(
    residual_recalibrated,
    solar
)[0, 1]

sqrt_rain = np.sqrt(
    np.maximum(rainfall, 0)
)

corr_sqrt_rain = np.corrcoef(
    residual_recalibrated,
    sqrt_rain
)[0, 1]


print("\n" + "-" * 80)
print("RESIDUAL STRUCTURE AFTER RECALIBRATION")
print("-" * 80)

print(f"Residual ↔ rainfall      = {corr_rain:.6f}")
print(f"Residual ↔ temperature   = {corr_temp:.6f}")
print(f"Residual ↔ solar         = {corr_solar:.6f}")
print(f"Residual ↔ sqrt(rainfall)= {corr_sqrt_rain:.6f}")


# ============================================================
# 10. SAVE
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "alpha",
        "MAE_M0",
        "RMSE_M0",
        "R2_M0",
        "MAE_Recalibrated",
        "RMSE_Recalibrated",
        "R2_Recalibrated",
        "MAE_Improvement",
        "RMSE_Improvement",
        "Residual_Rainfall",
        "Residual_Temperature",
        "Residual_Solar",
        "Residual_Sqrt_Rainfall"
    ],
    "Value": [
        alpha,
        mae_m0,
        rmse_m0,
        r2_m0,
        mae_recal,
        rmse_recal,
        r2_recal,
        mae_improvement,
        rmse_improvement,
        corr_rain,
        corr_temp,
        corr_solar,
        corr_sqrt_rain
    ]
})

results.to_csv(
    "step58_parameter_counterfactual_results.csv",
    index=False
)

print("\nSaved:")
print("step58_parameter_counterfactual_results.csv")

print("\n" + "=" * 80)
print("STEP 58 COMPLETE")
print("=" * 80)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 80)
print("ETMR — STEP 23: TEMPORAL STATE / LAG ABLATION")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

target = "sm_surface"

split_date = pd.Timestamp("2024-01-01")

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

y_train = train[target]
y_test = test[target]

# ---------------------------------------------------------
# 2. THREE FEATURE SETS
# ---------------------------------------------------------

full_features = [
    "temperature",
    "rainfall",
    "solar_radiation",
    "n_obs_3hourly",
    "ndvi",
    "rainfall_7d_sum",
    "solar_7d_mean",
    "sm_surface_lag1",
    "sm_rootzone_lag1",
    "ndvi_lag1"
]

no_surface_lag = [
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

no_soil_lags = [
    "temperature",
    "rainfall",
    "solar_radiation",
    "n_obs_3hourly",
    "ndvi",
    "rainfall_7d_sum",
    "solar_7d_mean",
    "ndvi_lag1"
]

experiments = {
    "Full_Clean_Model": full_features,
    "No_Surface_Lag": no_surface_lag,
    "No_Soil_Moisture_Lags": no_soil_lags
}

results = []

# ---------------------------------------------------------
# 3. TRAIN AND EVALUATE
# ---------------------------------------------------------

for name, features in experiments.items():

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=2026,
        n_jobs=-1
    )

    model.fit(
        train[features],
        y_train
    )

    pred = model.predict(test[features])

    residual = y_test.values - pred

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("Features:", len(features))
    print(f"MAE  : {mae:.9f}")
    print(f"RMSE : {rmse:.9f}")
    print(f"R²   : {r2:.9f}")
    print(f"Residual mean: {np.mean(residual):.9f}")
    print(f"Residual std : {np.std(residual):.9f}")

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Residual_Mean": np.mean(residual),
        "Residual_Std": np.std(residual)
    })

# ---------------------------------------------------------
# 4. RESULTS TABLE
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("ABLATION RESULTS")
print("=" * 80)

print(results_df.to_string(index=False))

# ---------------------------------------------------------
# 5. PERFORMANCE DROP
# ---------------------------------------------------------

full_r2 = results_df.loc[
    results_df["Model"] == "Full_Clean_Model",
    "R2"
].iloc[0]

no_surface_r2 = results_df.loc[
    results_df["Model"] == "No_Surface_Lag",
    "R2"
].iloc[0]

no_soil_r2 = results_df.loc[
    results_df["Model"] == "No_Soil_Moisture_Lags",
    "R2"
].iloc[0]

print("\n" + "=" * 80)
print("R² DROP RELATIVE TO FULL MODEL")
print("=" * 80)

print(
    f"No surface lag drop : "
    f"{full_r2 - no_surface_r2:.6f}"
)

print(
    f"No soil-moisture lag drop : "
    f"{full_r2 - no_soil_r2:.6f}"
)

# ---------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------

results_df.to_csv(
    "step23_lag_ablation_results.csv",
    index=False
)

print("\nSaved:")
print("step23_lag_ablation_results.csv")

print("\n" + "=" * 80)
print("STEP 23 COMPLETE")
print("=" * 80)
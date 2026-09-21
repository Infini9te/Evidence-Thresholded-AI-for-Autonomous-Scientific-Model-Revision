import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 80)
print("ETMR — STEP 22: CLEAN BASELINE AFTER LEAKAGE AUDIT")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------
# 2. TARGET
# ---------------------------------------------------------

target = "sm_surface"

# IMPORTANT:
# sm_surface_wetness is removed because it has correlation = 1.0
# with the target and therefore creates circular prediction.

features = [
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

# ---------------------------------------------------------
# 3. TEMPORAL SPLIT
# ---------------------------------------------------------

split_date = pd.Timestamp("2024-01-01")

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("\nDATA SPLIT")
print("-" * 80)

print("Total rows :", len(df))
print("Training   :", len(train))
print("Testing    :", len(test))

print(
    "Training period:",
    train["date"].min(),
    "→",
    train["date"].max()
)

print(
    "Testing period :",
    test["date"].min(),
    "→",
    test["date"].max()
)

print("\nFEATURES USED")
print("-" * 80)

for f in features:
    print("-", f)

print("\nREMOVED CIRCULAR FEATURE")
print("-" * 80)
print("sm_surface_wetness")

# ---------------------------------------------------------
# 4. MODEL
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=2026,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. PREDICTION
# ---------------------------------------------------------

pred = model.predict(X_test)

residual = y_test.values - pred

# ---------------------------------------------------------
# 6. METRICS
# ---------------------------------------------------------

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nMODEL PERFORMANCE")
print("-" * 80)

print(f"MAE  : {mae:.9f}")
print(f"RMSE : {rmse:.9f}")
print(f"R²   : {r2:.9f}")

print("\nRESIDUAL STATISTICS")
print("-" * 80)

print(f"Mean : {np.mean(residual):.9f}")
print(f"Std  : {np.std(residual):.9f}")
print(f"Min  : {np.min(residual):.9f}")
print(f"Max  : {np.max(residual):.9f}")

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("-" * 80)

print(importance.to_string(index=False))

# ---------------------------------------------------------
# 8. SAVE PREDICTIONS
# ---------------------------------------------------------

output = test[[
    "date",
    "sm_surface",
    "sm_surface_lag1",
    "sm_rootzone",
    "rainfall",
    "temperature",
    "solar_radiation",
    "ndvi"
]].copy()

output["prediction"] = pred
output["residual"] = residual

output.to_csv(
    "step22_clean_predictions.csv",
    index=False
)

print("\nSaved:")
print("step22_clean_predictions.csv")

print("\n" + "=" * 80)
print("STEP 22 COMPLETE")
print("=" * 80)
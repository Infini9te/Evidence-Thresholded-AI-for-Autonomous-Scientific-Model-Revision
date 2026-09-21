import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

features = [
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
]

target = "sm_surface"


# ============================================================
# 3. REMOVE INVALID ROWS
# ============================================================

data = df[["date"] + features + [target]].dropna().copy()


# ============================================================
# 4. TEMPORAL TRAIN / TEST SPLIT
# ============================================================

split_date = pd.Timestamp("2024-01-01")

train = data[data["date"] < split_date].copy()
test = data[data["date"] >= split_date].copy()


print("=" * 60)
print("TEMPORAL DATASET")
print("=" * 60)

print("Total:", len(data))
print("Training:", len(train))
print("Testing:", len(test))

print("\nTraining period:")
print(train["date"].min(), "→", train["date"].max())

print("\nTesting period:")
print(test["date"].min(), "→", test["date"].max())


# ============================================================
# 5. TRAIN BASELINE DIGITAL-TWIN SURROGATE
# ============================================================

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


model = RandomForestRegressor(
    n_estimators=300,
    random_state=2026,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# 6. PREDICTION
# ============================================================

test["prediction"] = model.predict(X_test)


# ============================================================
# 7. RESIDUAL
# ============================================================

test["residual"] = (
    test[target] - test["prediction"]
)


# ============================================================
# 8. STANDARD ERROR
# ============================================================

test["absolute_error"] = (
    test["residual"].abs()
)


# ============================================================
# 9. PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    y_test,
    test["prediction"]
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test["prediction"]
    )
)

r2 = r2_score(
    y_test,
    test["prediction"]
)


print("\n" + "=" * 60)
print("BASELINE PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")


# ============================================================
# 10. RESIDUAL STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("RESIDUAL EVIDENCE")
print("=" * 60)

print(
    test["residual"].describe()
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

test.to_csv(
    "step3_predictions.csv",
    index=False
)

print("\nSaved:")
print("step3_predictions.csv")
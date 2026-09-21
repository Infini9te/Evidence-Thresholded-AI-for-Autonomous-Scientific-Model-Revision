import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("=" * 80)
print("ETMR — STEP 60: BOOTSTRAP STRUCTURAL COUNTERFACTUAL")
print("=" * 80)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("final_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

target = "sm_surface"

base_features = [
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

# ============================================================
# 2. CHRONOLOGICAL SPLIT
# ============================================================

train = df[df["date"] < "2024-01-01"].copy()
test = df[df["date"] >= "2024-01-01"].copy()

# ============================================================
# 3. STRUCTURAL FEATURE
# ============================================================

train["sqrt_rainfall"] = np.sqrt(
    np.maximum(train["rainfall"], 0)
)

test["sqrt_rainfall"] = np.sqrt(
    np.maximum(test["rainfall"], 0)
)

# ============================================================
# 4. BASE MODEL M0
# ============================================================

X_train_m0 = train[base_features]
X_test_m0 = test[base_features]

y_train = train[target]
y_test = test[target]

model_m0 = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model_m0.fit(
    X_train_m0,
    y_train
)

pred_m0 = model_m0.predict(
    X_test_m0
)

# ============================================================
# 5. STRUCTURAL MODEL M1
# ============================================================

structural_features = base_features + [
    "sqrt_rainfall"
]

X_train_m1 = train[structural_features]
X_test_m1 = test[structural_features]

model_m1 = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model_m1.fit(
    X_train_m1,
    y_train
)

pred_m1 = model_m1.predict(
    X_test_m1
)

# ============================================================
# 6. OBSERVED METRICS
# ============================================================

y = y_test.values

mae_m0 = mean_absolute_error(
    y,
    pred_m0
)

mae_m1 = mean_absolute_error(
    y,
    pred_m1
)

rmse_m0 = np.sqrt(
    mean_squared_error(
        y,
        pred_m0
    )
)

rmse_m1 = np.sqrt(
    mean_squared_error(
        y,
        pred_m1
    )
)

observed_mae_gain = mae_m0 - mae_m1
observed_rmse_gain = rmse_m0 - rmse_m1

print("\nObserved results:")

print(f"MAE M0 = {mae_m0:.8f}")
print(f"MAE M1 = {mae_m1:.8f}")
print(f"MAE gain = {observed_mae_gain:.8f}")

print(f"\nRMSE M0 = {rmse_m0:.8f}")
print(f"RMSE M1 = {rmse_m1:.8f}")
print(f"RMSE gain = {observed_rmse_gain:.8f}")

# ============================================================
# 7. RESIDUALS
# ============================================================

residual_m0 = y - pred_m0
residual_m1 = y - pred_m1

rainfall = test["rainfall"].values

sqrt_rainfall = np.sqrt(
    np.maximum(rainfall, 0)
)

corr_m0 = np.corrcoef(
    residual_m0,
    sqrt_rainfall
)[0, 1]

corr_m1 = np.corrcoef(
    residual_m1,
    sqrt_rainfall
)[0, 1]

observed_structure_reduction = (
    abs(corr_m0) - abs(corr_m1)
)

print("\n" + "-" * 80)
print("STRUCTURAL RESIDUAL EVIDENCE")
print("-" * 80)

print(
    f"Residual ↔ sqrt(rainfall), M0 = "
    f"{corr_m0:.8f}"
)

print(
    f"Residual ↔ sqrt(rainfall), M1 = "
    f"{corr_m1:.8f}"
)

print(
    f"Structure reduction = "
    f"{observed_structure_reduction:.8f}"
)

# ============================================================
# 8. BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    20260911
)

N_BOOT = 2000
n = len(y)

mae_gains = []
rmse_gains = []
structure_reductions = []

for i in range(N_BOOT):

    idx = rng.integers(
        0,
        n,
        size=n
    )

    y_b = y[idx]

    pred0_b = pred_m0[idx]
    pred1_b = pred_m1[idx]

    sqrt_rain_b = sqrt_rainfall[idx]

    mae0 = np.mean(
        np.abs(
            y_b - pred0_b
        )
    )

    mae1 = np.mean(
        np.abs(
            y_b - pred1_b
        )
    )

    rmse0 = np.sqrt(
        np.mean(
            (y_b - pred0_b) ** 2
        )
    )

    rmse1 = np.sqrt(
        np.mean(
            (y_b - pred1_b) ** 2
        )
    )

    corr0 = np.corrcoef(
        y_b - pred0_b,
        sqrt_rain_b
    )[0, 1]

    corr1 = np.corrcoef(
        y_b - pred1_b,
        sqrt_rain_b
    )[0, 1]

    mae_gains.append(
        mae0 - mae1
    )

    rmse_gains.append(
        rmse0 - rmse1
    )

    structure_reductions.append(
        abs(corr0) - abs(corr1)
    )

mae_gains = np.array(
    mae_gains
)

rmse_gains = np.array(
    rmse_gains
)

structure_reductions = np.array(
    structure_reductions
)

# ============================================================
# 9. CONFIDENCE INTERVALS
# ============================================================

mae_ci = np.percentile(
    mae_gains,
    [2.5, 97.5]
)

rmse_ci = np.percentile(
    rmse_gains,
    [2.5, 97.5]
)

structure_ci = np.percentile(
    structure_reductions,
    [2.5, 97.5]
)

p_mae = np.mean(
    mae_gains > 0
)

p_rmse = np.mean(
    rmse_gains > 0
)

p_structure = np.mean(
    structure_reductions > 0
)

# ============================================================
# 10. PRINT BOOTSTRAP RESULTS
# ============================================================

print("\n" + "-" * 80)
print("BOOTSTRAP STRUCTURAL EVIDENCE")
print("-" * 80)

print("\nMAE gain:")
print(
    f"Mean = {mae_gains.mean():.8f}"
)

print(
    f"95% CI = "
    f"[{mae_ci[0]:.8f}, "
    f"{mae_ci[1]:.8f}]"
)

print(
    f"P(MAE improvement > 0) = "
    f"{p_mae:.4f}"
)

print("\nRMSE gain:")
print(
    f"Mean = {rmse_gains.mean():.8f}"
)

print(
    f"95% CI = "
    f"[{rmse_ci[0]:.8f}, "
    f"{rmse_ci[1]:.8f}]"
)

print(
    f"P(RMSE improvement > 0) = "
    f"{p_rmse:.4f}"
)

print("\nResidual structural reduction:")
print(
    f"Mean = "
    f"{structure_reductions.mean():.8f}"
)

print(
    f"95% CI = "
    f"[{structure_ci[0]:.8f}, "
    f"{structure_ci[1]:.8f}]"
)

print(
    f"P(structural reduction > 0) = "
    f"{p_structure:.4f}"
)

# ============================================================
# 11. CONSERVATIVE INTERPRETATION
# ============================================================

print("\n" + "-" * 80)
print("STRUCTURAL COUNTERFACTUAL INTERPRETATION")
print("-" * 80)

if (
    rmse_ci[0] > 0
    and p_rmse >= 0.95
):
    rmse_evidence = "ROBUST"
else:
    rmse_evidence = "WEAK_OR_INCONCLUSIVE"

if (
    mae_ci[0] > 0
    and p_mae >= 0.95
):
    mae_evidence = "ROBUST"
else:
    mae_evidence = "WEAK_OR_INCONCLUSIVE"

if (
    structure_ci[0] > 0
    and p_structure >= 0.95
):
    structural_evidence = "ROBUST"
else:
    structural_evidence = "WEAK_OR_INCONCLUSIVE"

print(
    "MAE evidence       :", 
    mae_evidence
)

print(
    "RMSE evidence      :", 
    rmse_evidence
)

print(
    "Structural evidence:",
    structural_evidence
)

if (
    rmse_evidence == "ROBUST"
    and structural_evidence == "ROBUST"
):
    interpretation = (
        "STRUCTURAL_COUNTERFACTUAL_SUPPORTED"
    )
else:
    interpretation = (
        "STRUCTURAL_COUNTERFACTUAL_NOT_STRONGLY_SUPPORTED"
    )

print("\nFinal interpretation:")
print(interpretation)

# ============================================================
# 12. SAVE
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Observed_MAE_Gain",
        "Bootstrap_Mean_MAE_Gain",
        "MAE_CI_Lower",
        "MAE_CI_Upper",
        "P_MAE_Gain_Positive",
        "Observed_RMSE_Gain",
        "Bootstrap_Mean_RMSE_Gain",
        "RMSE_CI_Lower",
        "RMSE_CI_Upper",
        "P_RMSE_Gain_Positive",
        "Observed_Structure_Reduction",
        "Bootstrap_Mean_Structure_Reduction",
        "Structure_CI_Lower",
        "Structure_CI_Upper",
        "P_Structure_Reduction_Positive"
    ],
    "Value": [
        observed_mae_gain,
        mae_gains.mean(),
        mae_ci[0],
        mae_ci[1],
        p_mae,
        observed_rmse_gain,
        rmse_gains.mean(),
        rmse_ci[0],
        rmse_ci[1],
        p_rmse,
        observed_structure_reduction,
        structure_reductions.mean(),
        structure_ci[0],
        structure_ci[1],
        p_structure
    ]
})

summary.to_csv(
    "step60_bootstrap_structural_counterfactual_results.csv",
    index=False
)

print("\nSaved:")
print(
    "step60_bootstrap_structural_counterfactual_results.csv"
)

print("\n" + "=" * 80)
print("STEP 60 COMPLETE")
print("=" * 80)
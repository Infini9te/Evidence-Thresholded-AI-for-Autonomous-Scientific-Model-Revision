import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("=" * 80)
print("ETMR — STEP 59: BOOTSTRAP PARAMETER COUNTERFACTUAL")
print("=" * 80)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("final_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

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

train = df[df["date"] < "2024-01-01"].copy()
test = df[df["date"] >= "2024-01-01"].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# ============================================================
# 2. FIT BASE MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# ============================================================
# 3. ESTIMATE PARAMETER CORRECTION FROM TRAINING ONLY
# ============================================================

train_mean = y_train.mean()

numerator = np.sum(
    (train_pred - train_mean)
    * (y_train.values - train_mean)
)

denominator = np.sum(
    (train_pred - train_mean) ** 2
)

alpha = numerator / denominator

test_pred_recal = (
    train_mean
    + alpha * (test_pred - train_mean)
)

# ============================================================
# 4. TEST-SET RESIDUALS
# ============================================================

y = y_test.values

residual_m0 = y - test_pred
residual_recal = y - test_pred_recal

# ============================================================
# 5. OBSERVED IMPROVEMENT
# ============================================================

mae_m0 = mean_absolute_error(y, test_pred)
mae_recal = mean_absolute_error(y, test_pred_recal)

rmse_m0 = np.sqrt(
    mean_squared_error(y, test_pred)
)

rmse_recal = np.sqrt(
    mean_squared_error(y, test_pred_recal)
)

observed_mae_gain = mae_m0 - mae_recal
observed_rmse_gain = rmse_m0 - rmse_recal

print("\nObserved results:")
print(f"Alpha              = {alpha:.8f}")
print(f"MAE M0             = {mae_m0:.8f}")
print(f"MAE recalibrated   = {mae_recal:.8f}")
print(f"MAE gain           = {observed_mae_gain:.8f}")

print(f"\nRMSE M0            = {rmse_m0:.8f}")
print(f"RMSE recalibrated  = {rmse_recal:.8f}")
print(f"RMSE gain          = {observed_rmse_gain:.8f}")

# ============================================================
# 6. BOOTSTRAP TEST INSTANCES
# ============================================================

rng = np.random.default_rng(20260911)

N_BOOT = 2000
n = len(y)

mae_gains = []
rmse_gains = []

for i in range(N_BOOT):

    idx = rng.integers(
        0,
        n,
        size=n
    )

    y_b = y[idx]
    pred_m0_b = test_pred[idx]
    pred_recal_b = test_pred_recal[idx]

    mae0 = np.mean(
        np.abs(y_b - pred_m0_b)
    )

    maer = np.mean(
        np.abs(y_b - pred_recal_b)
    )

    rmse0 = np.sqrt(
        np.mean(
            (y_b - pred_m0_b) ** 2
        )
    )

    rmser = np.sqrt(
        np.mean(
            (y_b - pred_recal_b) ** 2
        )
    )

    mae_gains.append(
        mae0 - maer
    )

    rmse_gains.append(
        rmse0 - rmser
    )

mae_gains = np.array(mae_gains)
rmse_gains = np.array(rmse_gains)

# ============================================================
# 7. CONFIDENCE INTERVALS
# ============================================================

mae_ci = np.percentile(
    mae_gains,
    [2.5, 97.5]
)

rmse_ci = np.percentile(
    rmse_gains,
    [2.5, 97.5]
)

prob_mae_positive = np.mean(
    mae_gains > 0
)

prob_rmse_positive = np.mean(
    rmse_gains > 0
)

# ============================================================
# 8. PRINT
# ============================================================

print("\n" + "-" * 80)
print("BOOTSTRAP EVIDENCE")
print("-" * 80)

print("\nMAE gain:")
print(f"Mean = {mae_gains.mean():.8f}")
print(
    f"95% CI = [{mae_ci[0]:.8f}, "
    f"{mae_ci[1]:.8f}]"
)
print(
    f"P(MAE improvement > 0) = "
    f"{prob_mae_positive:.4f}"
)

print("\nRMSE gain:")
print(f"Mean = {rmse_gains.mean():.8f}")
print(
    f"95% CI = [{rmse_ci[0]:.8f}, "
    f"{rmse_ci[1]:.8f}]"
)
print(
    f"P(RMSE improvement > 0) = "
    f"{prob_rmse_positive:.4f}"
)

# ============================================================
# 9. ETMR PARAMETER EVIDENCE
# ============================================================

print("\n" + "-" * 80)
print("PARAMETER COUNTERFACTUAL INTERPRETATION")
print("-" * 80)

if (
    rmse_ci[0] > 0
    and prob_rmse_positive >= 0.95
):
    rmse_evidence = "ROBUST"
else:
    rmse_evidence = "WEAK_OR_INCONCLUSIVE"

if (
    mae_ci[0] > 0
    and prob_mae_positive >= 0.95
):
    mae_evidence = "ROBUST"
else:
    mae_evidence = "WEAK_OR_INCONCLUSIVE"

print("MAE evidence :", mae_evidence)
print("RMSE evidence:", rmse_evidence)

if (
    rmse_evidence == "ROBUST"
    and mae_evidence == "ROBUST"
):
    interpretation = "PARAMETER_CORRECTION_SUPPORTED"
else:
    interpretation = "PARAMETER_CORRECTION_NOT_STRONGLY_SUPPORTED"

print("\nFinal interpretation:")
print(interpretation)

# ============================================================
# 10. SAVE
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "alpha",
        "Observed_MAE_Gain",
        "Bootstrap_Mean_MAE_Gain",
        "MAE_CI_Lower",
        "MAE_CI_Upper",
        "P_MAE_Gain_Positive",
        "Observed_RMSE_Gain",
        "Bootstrap_Mean_RMSE_Gain",
        "RMSE_CI_Lower",
        "RMSE_CI_Upper",
        "P_RMSE_Gain_Positive"
    ],
    "Value": [
        alpha,
        observed_mae_gain,
        mae_gains.mean(),
        mae_ci[0],
        mae_ci[1],
        prob_mae_positive,
        observed_rmse_gain,
        rmse_gains.mean(),
        rmse_ci[0],
        rmse_ci[1],
        prob_rmse_positive
    ]
})

summary.to_csv(
    "step59_bootstrap_parameter_counterfactual_results.csv",
    index=False
)

print("\nSaved:")
print(
    "step59_bootstrap_parameter_counterfactual_results.csv"
)

print("\n" + "=" * 80)
print("STEP 59 COMPLETE")
print("=" * 80)
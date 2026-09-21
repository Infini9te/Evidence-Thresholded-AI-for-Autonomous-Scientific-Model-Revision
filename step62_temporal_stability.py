import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("=" * 80)
print("ETMR — STEP 62: TEMPORAL STABILITY OF COUNTERFACTUAL DECISION")
print("=" * 80)

# ============================================================
# 1. LOAD DATA
# ============================================================

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

# ============================================================
# 2. TEMPORAL WINDOWS
# ============================================================

windows = {
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01")
}

all_results = []

# ============================================================
# 3. LOOP THROUGH WINDOWS
# ============================================================

for year, (start, end) in windows.items():

    print("\n" + "=" * 80)
    print(f"EVALUATION YEAR: {year}")
    print("=" * 80)

    # --------------------------------------------------------
    # Expanding-window training
    # --------------------------------------------------------

    train = df[
        df["date"] < start
    ].copy()

    test = df[
        (df["date"] >= start)
        & (df["date"] < end)
    ].copy()

    print(
        f"Train samples = {len(train)}"
    )

    print(
        f"Test samples  = {len(test)}"
    )

    if len(test) == 0:
        print("No test data. Skipping.")
        continue

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    # --------------------------------------------------------
    # M0
    # --------------------------------------------------------

    m0 = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    m0.fit(
        X_train,
        y_train
    )

    pred_m0_train = m0.predict(
        X_train
    )

    pred_m0 = m0.predict(
        X_test
    )

    # --------------------------------------------------------
    # Parameter recalibration
    # --------------------------------------------------------

    train_mean = y_train.mean()

    numerator = np.sum(
        (pred_m0_train - train_mean)
        * (y_train.values - train_mean)
    )

    denominator = np.sum(
        (pred_m0_train - train_mean) ** 2
    )

    alpha = numerator / denominator

    pred_recal = (
        train_mean
        + alpha * (pred_m0 - train_mean)
    )

    # --------------------------------------------------------
    # M1 structural model
    # --------------------------------------------------------

    train_m1 = train.copy()
    test_m1 = test.copy()

    train_m1["sqrt_rainfall"] = np.sqrt(
        np.maximum(
            train_m1["rainfall"],
            0
        )
    )

    test_m1["sqrt_rainfall"] = np.sqrt(
        np.maximum(
            test_m1["rainfall"],
            0
        )
    )

    structural_features = features + [
        "sqrt_rainfall"
    ]

    m1 = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    m1.fit(
        train_m1[structural_features],
        y_train
    )

    pred_m1 = m1.predict(
        test_m1[structural_features]
    )

    # ========================================================
    # METRICS
    # ========================================================

    y = y_test.values

    mae_m0 = mean_absolute_error(
        y,
        pred_m0
    )

    rmse_m0 = np.sqrt(
        mean_squared_error(
            y,
            pred_m0
        )
    )

    mae_recal = mean_absolute_error(
        y,
        pred_recal
    )

    rmse_recal = np.sqrt(
        mean_squared_error(
            y,
            pred_recal
        )
    )

    mae_m1 = mean_absolute_error(
        y,
        pred_m1
    )

    rmse_m1 = np.sqrt(
        mean_squared_error(
            y,
            pred_m1
        )
    )

    # ========================================================
    # IMPROVEMENTS
    # ========================================================

    parameter_mae_gain = (
        mae_m0 - mae_recal
    )

    parameter_rmse_gain = (
        rmse_m0 - rmse_recal
    )

    structural_mae_gain = (
        mae_m0 - mae_m1
    )

    structural_rmse_gain = (
        rmse_m0 - rmse_m1
    )

    # ========================================================
    # RESIDUAL STRUCTURE
    # ========================================================

    sqrt_rainfall = np.sqrt(
        np.maximum(
            test["rainfall"].values,
            0
        )
    )

    residual_m0 = (
        y - pred_m0
    )

    residual_m1 = (
        y - pred_m1
    )

    corr_m0 = np.corrcoef(
        residual_m0,
        sqrt_rainfall
    )[0, 1]

    corr_m1 = np.corrcoef(
        residual_m1,
        sqrt_rainfall
    )[0, 1]

    structure_reduction = (
        abs(corr_m0)
        - abs(corr_m1)
    )

    # ========================================================
    # CONSERVATIVE ETMR DECISION
    # ========================================================

    # We intentionally DO NOT use newly tuned thresholds.
    #
    # Strong evidence requirements:
    #
    # Parameter:
    # robust improvement would be needed.
    #
    # Structural:
    # predictive improvement AND removal of residual structure.
    #
    # With only one temporal window we cannot establish
    # bootstrap robustness here, so we label the result
    # based on directional evidence only.

    if (
        structural_rmse_gain > 0
        and structure_reduction > 0
    ):
        directional_structural = "SUPPORTED"

    else:
        directional_structural = "NOT_SUPPORTED"

    if (
        parameter_rmse_gain > 0
        and parameter_mae_gain > 0
    ):
        directional_parameter = "SUPPORTED"

    else:
        directional_parameter = "NOT_SUPPORTED"

    if (
        directional_structural == "SUPPORTED"
    ):
        decision = "REVISE"

    elif (
        directional_parameter == "SUPPORTED"
    ):
        decision = "RECALIBRATE"

    else:
        decision = "KEEP"

    # ========================================================
    # PRINT
    # ========================================================

    print("\nM0")
    print(
        f"MAE  = {mae_m0:.8f}"
    )
    print(
        f"RMSE = {rmse_m0:.8f}"
    )

    print("\nParameter recalibration")
    print(
        f"Alpha = {alpha:.8f}"
    )
    print(
        f"MAE  = {mae_recal:.8f}"
    )
    print(
        f"RMSE = {rmse_recal:.8f}"
    )

    print(
        f"MAE gain  = "
        f"{parameter_mae_gain:.8f}"
    )

    print(
        f"RMSE gain = "
        f"{parameter_rmse_gain:.8f}"
    )

    print("\nStructural M1")
    print(
        f"MAE  = {mae_m1:.8f}"
    )
    print(
        f"RMSE = {rmse_m1:.8f}"
    )

    print(
        f"MAE gain  = "
        f"{structural_mae_gain:.8f}"
    )

    print(
        f"RMSE gain = "
        f"{structural_rmse_gain:.8f}"
    )

    print("\nResidual structure")

    print(
        f"M0 residual ↔ sqrt(rainfall) = "
        f"{corr_m0:.6f}"
    )

    print(
        f"M1 residual ↔ sqrt(rainfall) = "
        f"{corr_m1:.6f}"
    )

    print(
        f"Structure reduction = "
        f"{structure_reduction:.6f}"
    )

    print(
        "\nDirectional parameter evidence:",
        directional_parameter
    )

    print(
        "Directional structural evidence:",
        directional_structural
    )

    print(
        "\nETMR decision:",
        decision
    )

    # ========================================================
    # STORE
    # ========================================================

    all_results.append({
        "Year": year,
        "Train_N": len(train),
        "Test_N": len(test),

        "Alpha": alpha,

        "MAE_M0": mae_m0,
        "RMSE_M0": rmse_m0,

        "MAE_Recalibrated": mae_recal,
        "RMSE_Recalibrated": rmse_recal,

        "Parameter_MAE_Gain":
            parameter_mae_gain,

        "Parameter_RMSE_Gain":
            parameter_rmse_gain,

        "MAE_M1": mae_m1,
        "RMSE_M1": rmse_m1,

        "Structural_MAE_Gain":
            structural_mae_gain,

        "Structural_RMSE_Gain":
            structural_rmse_gain,

        "M0_SqrtRain_Corr":
            corr_m0,

        "M1_SqrtRain_Corr":
            corr_m1,

        "Structural_Reduction":
            structure_reduction,

        "Parameter_Evidence":
            directional_parameter,

        "Structural_Evidence":
            directional_structural,

        "ETMR_Decision":
            decision
    })


# ============================================================
# 4. SUMMARY
# ============================================================

results = pd.DataFrame(
    all_results
)

print("\n" + "=" * 80)
print("TEMPORAL STABILITY SUMMARY")
print("=" * 80)

print(
    results[
        [
            "Year",
            "Parameter_RMSE_Gain",
            "Structural_RMSE_Gain",
            "Structural_Reduction",
            "ETMR_Decision"
        ]
    ].to_string(index=False)
)

print("\nDecision counts:")

print(
    results["ETMR_Decision"]
    .value_counts()
)

# ============================================================
# 5. SAVE
# ============================================================

results.to_csv(
    "step62_temporal_stability_results.csv",
    index=False
)

print("\nSaved:")
print(
    "step62_temporal_stability_results.csv"
)

print("\n" + "=" * 80)
print("STEP 62 COMPLETE")
print("=" * 80)
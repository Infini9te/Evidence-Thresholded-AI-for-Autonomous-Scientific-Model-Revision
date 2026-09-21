import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("=" * 80)
print("ETMR — STEP 63: MULTI-PERIOD BOOTSTRAP VALIDATION")
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

windows = {
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01")
}

N_BOOT = 2000

rng = np.random.default_rng(20260911)

all_results = []

# ============================================================
# 2. LOOP THROUGH YEARS
# ============================================================

for year, (start, end) in windows.items():

    print("\n" + "=" * 80)
    print(f"BOOTSTRAP PERIOD: {year}")
    print("=" * 80)

    train = df[
        df["date"] < start
    ].copy()

    test = df[
        (df["date"] >= start)
        & (df["date"] < end)
    ].copy()

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

    train_pred = m0.predict(
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
        (train_pred - train_mean)
        * (y_train.values - train_mean)
    )

    denominator = np.sum(
        (train_pred - train_mean) ** 2
    )

    alpha = numerator / denominator

    pred_recal = (
        train_mean
        + alpha *
        (pred_m0 - train_mean)
    )

    # --------------------------------------------------------
    # Structural M1
    # --------------------------------------------------------

    train["sqrt_rainfall"] = np.sqrt(
        np.maximum(
            train["rainfall"],
            0
        )
    )

    test["sqrt_rainfall"] = np.sqrt(
        np.maximum(
            test["rainfall"],
            0
        )
    )

    structural_features = (
        features +
        ["sqrt_rainfall"]
    )

    m1 = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    m1.fit(
        train[structural_features],
        y_train
    )

    pred_m1 = m1.predict(
        test[structural_features]
    )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    y = y_test.values

    sqrt_rainfall = test[
        "sqrt_rainfall"
    ].values

    residual_m0 = (
        y - pred_m0
    )

    residual_m1 = (
        y - pred_m1
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    parameter_mae = []
    parameter_rmse = []

    structural_mae = []
    structural_rmse = []

    structural_reduction = []

    n = len(y)

    for b in range(N_BOOT):

        idx = rng.integers(
            0,
            n,
            size=n
        )

        y_b = y[idx]

        m0_b = pred_m0[idx]
        recal_b = pred_recal[idx]
        m1_b = pred_m1[idx]

        rain_b = sqrt_rainfall[idx]

        # Parameter

        mae0 = np.mean(
            np.abs(
                y_b - m0_b
            )
        )

        maer = np.mean(
            np.abs(
                y_b - recal_b
            )
        )

        rmse0 = np.sqrt(
            np.mean(
                (y_b - m0_b) ** 2
            )
        )

        rmser = np.sqrt(
            np.mean(
                (y_b - recal_b) ** 2
            )
        )

        parameter_mae.append(
            mae0 - maer
        )

        parameter_rmse.append(
            rmse0 - rmser
        )

        # Structural

        mae1 = np.mean(
            np.abs(
                y_b - m1_b
            )
        )

        rmse1 = np.sqrt(
            np.mean(
                (y_b - m1_b) ** 2
            )
        )

        structural_mae.append(
            mae0 - mae1
        )

        structural_rmse.append(
            rmse0 - rmse1
        )

        # Residual structure

        r0 = y_b - m0_b
        r1 = y_b - m1_b

        corr0 = np.corrcoef(
            r0,
            rain_b
        )[0, 1]

        corr1 = np.corrcoef(
            r1,
            rain_b
        )[0, 1]

        structural_reduction.append(
            abs(corr0) -
            abs(corr1)
        )

    # Convert

    parameter_mae = np.array(
        parameter_mae
    )

    parameter_rmse = np.array(
        parameter_rmse
    )

    structural_mae = np.array(
        structural_mae
    )

    structural_rmse = np.array(
        structural_rmse
    )

    structural_reduction = np.array(
        structural_reduction
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summarize(values):

        ci = np.percentile(
            values,
            [2.5, 97.5]
        )

        return (
            values.mean(),
            ci[0],
            ci[1],
            np.mean(values > 0)
        )

    p_mae = summarize(
        parameter_mae
    )

    p_rmse = summarize(
        parameter_rmse
    )

    s_mae = summarize(
        structural_mae
    )

    s_rmse = summarize(
        structural_rmse
    )

    s_structure = summarize(
        structural_reduction
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\nParameter RMSE:"
    )

    print(
        f"Mean = {p_rmse[0]:.8f}"
    )

    print(
        f"95% CI = "
        f"[{p_rmse[1]:.8f}, "
        f"{p_rmse[2]:.8f}]"
    )

    print(
        f"P(>0) = {p_rmse[3]:.4f}"
    )

    print(
        "\nStructural RMSE:"
    )

    print(
        f"Mean = {s_rmse[0]:.8f}"
    )

    print(
        f"95% CI = "
        f"[{s_rmse[1]:.8f}, "
        f"{s_rmse[2]:.8f}]"
    )

    print(
        f"P(>0) = {s_rmse[3]:.4f}"
    )

    print(
        "\nStructural residual reduction:"
    )

    print(
        f"Mean = {s_structure[0]:.8f}"
    )

    print(
        f"95% CI = "
        f"[{s_structure[1]:.8f}, "
        f"{s_structure[2]:.8f}]"
    )

    print(
        f"P(>0) = {s_structure[3]:.4f}"
    )

    # ========================================================
    # STORE
    # ========================================================

    all_results.append({

        "Year": year,

        "Alpha": alpha,

        "Parameter_MAE_Mean":
            p_mae[0],

        "Parameter_MAE_CI_Lower":
            p_mae[1],

        "Parameter_MAE_CI_Upper":
            p_mae[2],

        "Parameter_MAE_P":
            p_mae[3],

        "Parameter_RMSE_Mean":
            p_rmse[0],

        "Parameter_RMSE_CI_Lower":
            p_rmse[1],

        "Parameter_RMSE_CI_Upper":
            p_rmse[2],

        "Parameter_RMSE_P":
            p_rmse[3],

        "Structural_MAE_Mean":
            s_mae[0],

        "Structural_MAE_CI_Lower":
            s_mae[1],

        "Structural_MAE_CI_Upper":
            s_mae[2],

        "Structural_MAE_P":
            s_mae[3],

        "Structural_RMSE_Mean":
            s_rmse[0],

        "Structural_RMSE_CI_Lower":
            s_rmse[1],

        "Structural_RMSE_CI_Upper":
            s_rmse[2],

        "Structural_RMSE_P":
            s_rmse[3],

        "Structural_Reduction_Mean":
            s_structure[0],

        "Structural_Reduction_CI_Lower":
            s_structure[1],

        "Structural_Reduction_CI_Upper":
            s_structure[2],

        "Structural_Reduction_P":
            s_structure[3]
    })


# ============================================================
# 3. SAVE
# ============================================================

results = pd.DataFrame(
    all_results
)

print("\n" + "=" * 80)
print("MULTI-PERIOD SUMMARY")
print("=" * 80)

print(
    results[
        [
            "Year",
            "Parameter_RMSE_P",
            "Structural_RMSE_P",
            "Structural_Reduction_P"
        ]
    ].to_string(index=False)
)

results.to_csv(
    "step63_multiperiod_bootstrap_results.csv",
    index=False
)

print("\nSaved:")
print(
    "step63_multiperiod_bootstrap_results.csv"
)

print("\n" + "=" * 80)
print("STEP 63 COMPLETE")
print("=" * 80)
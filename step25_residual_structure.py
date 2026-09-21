import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr

print("=" * 80)
print("ETMR — STEP 25: RESIDUAL STRUCTURE TEST")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD RESULTS FROM STEP 24
# ---------------------------------------------------------

df = pd.read_csv(
    "step24_competing_models_predictions.csv"
)

df["rainfall_nonlinear"] = np.sqrt(
    np.maximum(df["rainfall"], 0)
)

# ---------------------------------------------------------
# 2. RESIDUAL CORRELATIONS
# ---------------------------------------------------------

print("\nRESIDUAL ↔ ENVIRONMENT RELATIONSHIPS")
print("-" * 80)

variables = {
    "rainfall": df["rainfall"],
    "sqrt_rainfall": df["rainfall_nonlinear"],
    "temperature": df["temperature"],
    "solar_radiation": df["solar_radiation"]
}

for name, x in variables.items():

    m0_valid = (
        np.isfinite(df["M0_residual"]) &
        np.isfinite(x)
    )

    m1_valid = (
        np.isfinite(df["M1_residual"]) &
        np.isfinite(x)
    )

    m0_corr, m0_p = pearsonr(
        df.loc[m0_valid, "M0_residual"],
        x[m0_valid]
    )

    m1_corr, m1_p = pearsonr(
        df.loc[m1_valid, "M1_residual"],
        x[m1_valid]
    )

    print(f"\n{name}")

    print(
        f"M0 Pearson correlation : "
        f"{m0_corr:.6f}"
    )

    print(
        f"M0 p-value             : "
        f"{m0_p:.6e}"
    )

    print(
        f"M1 Pearson correlation : "
        f"{m1_corr:.6f}"
    )

    print(
        f"M1 p-value             : "
        f"{m1_p:.6e}"
    )

# ---------------------------------------------------------
# 3. SPEARMAN TEST
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("NONLINEAR / RANK-BASED TEST")
print("=" * 80)

x = df["rainfall_nonlinear"]

m0_corr, m0_p = spearmanr(
    df["M0_residual"],
    x
)

m1_corr, m1_p = spearmanr(
    df["M1_residual"],
    x
)

print(
    f"M0 Spearman correlation : {m0_corr:.6f}"
)

print(
    f"M0 Spearman p-value     : {m0_p:.6e}"
)

print(
    f"M1 Spearman correlation : {m1_corr:.6f}"
)

print(
    f"M1 Spearman p-value     : {m1_p:.6e}"
)

# ---------------------------------------------------------
# 4. RAINFALL REGIMES
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("RESIDUAL BIAS BY RAINFALL REGIME")
print("=" * 80)

df["rainfall_regime"] = pd.qcut(
    df["rainfall"],
    q=4,
    labels=[
        "Low",
        "Moderate",
        "High",
        "Extreme"
    ],
    duplicates="drop"
)

regime_results = []

for regime, group in df.groupby(
    "rainfall_regime",
    observed=False
):

    m0_mean = group["M0_residual"].mean()
    m1_mean = group["M1_residual"].mean()

    m0_mae = group["M0_residual"].abs().mean()
    m1_mae = group["M1_residual"].abs().mean()

    print(f"\n{regime}")

    print(
        f"M0 residual mean : {m0_mean:.8f}"
    )

    print(
        f"M1 residual mean : {m1_mean:.8f}"
    )

    print(
        f"M0 MAE           : {m0_mae:.8f}"
    )

    print(
        f"M1 MAE           : {m1_mae:.8f}"
    )

    regime_results.append({
        "Rainfall_Regime": str(regime),
        "M0_Residual_Mean": m0_mean,
        "M1_Residual_Mean": m1_mean,
        "M0_MAE": m0_mae,
        "M1_MAE": m1_mae,
        "Samples": len(group)
    })

# ---------------------------------------------------------
# 5. OVERALL RESIDUAL STRUCTURE REDUCTION
# ---------------------------------------------------------

m0_rain_corr = abs(
    pearsonr(
        df["M0_residual"],
        df["rainfall_nonlinear"]
    )[0]
)

m1_rain_corr = abs(
    pearsonr(
        df["M1_residual"],
        df["rainfall_nonlinear"]
    )[0]
)

print("\n" + "=" * 80)
print("STRUCTURAL RESIDUAL REDUCTION")
print("=" * 80)

print(
    f"|M0 residual ↔ sqrt(rainfall)| : "
    f"{m0_rain_corr:.6f}"
)

print(
    f"|M1 residual ↔ sqrt(rainfall)| : "
    f"{m1_rain_corr:.6f}"
)

print(
    f"Correlation reduction            : "
    f"{m0_rain_corr - m1_rain_corr:.6f}"
)

if m0_rain_corr > 0:

    reduction_pct = (
        (m0_rain_corr - m1_rain_corr)
        / m0_rain_corr
    ) * 100

    print(
        f"Reduction percentage             : "
        f"{reduction_pct:.2f}%"
    )

# ---------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------

regime_df = pd.DataFrame(regime_results)

regime_df.to_csv(
    "step25_residual_regimes.csv",
    index=False
)

print("\nSaved:")
print("step25_residual_regimes.csv")

print("\n" + "=" * 80)
print("STEP 25 COMPLETE")
print("=" * 80)
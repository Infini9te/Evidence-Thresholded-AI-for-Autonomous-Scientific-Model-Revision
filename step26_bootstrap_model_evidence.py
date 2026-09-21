import pandas as pd
import numpy as np

from scipy.stats import pearsonr

print("=" * 80)
print("ETMR — STEP 26: BOOTSTRAPPED MODEL-COMPARISON EVIDENCE")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD STEP 24 RESULTS
# ---------------------------------------------------------

df = pd.read_csv(
    "step24_competing_models_predictions.csv"
)

df["sqrt_rainfall"] = np.sqrt(
    np.maximum(df["rainfall"], 0)
)

# ---------------------------------------------------------
# 2. OBSERVED MODEL ERRORS
# ---------------------------------------------------------

m0_error = df["M0_residual"].values
m1_error = df["M1_residual"].values

m0_abs = np.abs(m0_error)
m1_abs = np.abs(m1_error)

# ---------------------------------------------------------
# 3. OBSERVED IMPROVEMENT
# ---------------------------------------------------------

m0_mae = np.mean(m0_abs)
m1_mae = np.mean(m1_abs)

mae_improvement = m0_mae - m1_mae

m0_rmse = np.sqrt(np.mean(m0_error ** 2))
m1_rmse = np.sqrt(np.mean(m1_error ** 2))

rmse_improvement = m0_rmse - m1_rmse

m0_corr = abs(
    pearsonr(
        m0_error,
        df["sqrt_rainfall"]
    )[0]
)

m1_corr = abs(
    pearsonr(
        m1_error,
        df["sqrt_rainfall"]
    )[0]
)

structure_reduction = m0_corr - m1_corr

print("\nOBSERVED MODEL DIFFERENCE")
print("-" * 80)

print(f"M0 MAE : {m0_mae:.9f}")
print(f"M1 MAE : {m1_mae:.9f}")

print(f"MAE improvement : {mae_improvement:.9f}")

print()

print(f"M0 RMSE : {m0_rmse:.9f}")
print(f"M1 RMSE : {m1_rmse:.9f}")

print(f"RMSE improvement : {rmse_improvement:.9f}")

print()

print(
    f"M0 |residual ↔ sqrt(rainfall)| : "
    f"{m0_corr:.6f}"
)

print(
    f"M1 |residual ↔ sqrt(rainfall)| : "
    f"{m1_corr:.6f}"
)

print(
    f"Residual structure reduction : "
    f"{structure_reduction:.6f}"
)

# ---------------------------------------------------------
# 4. BOOTSTRAP
# ---------------------------------------------------------

rng = np.random.default_rng(2026)

n = len(df)
n_bootstrap = 2000

mae_differences = []
rmse_differences = []
structure_differences = []

for i in range(n_bootstrap):

    indices = rng.integers(
        0,
        n,
        size=n
    )

    m0_sample = m0_error[indices]
    m1_sample = m1_error[indices]

    rainfall_sample = df[
        "sqrt_rainfall"
    ].values[indices]

    # MAE difference
    mae_diff = (
        np.mean(np.abs(m0_sample))
        -
        np.mean(np.abs(m1_sample))
    )

    # RMSE difference
    rmse_diff = (
        np.sqrt(np.mean(m0_sample ** 2))
        -
        np.sqrt(np.mean(m1_sample ** 2))
    )

    # Residual structure difference
    corr0 = abs(
        pearsonr(
            m0_sample,
            rainfall_sample
        )[0]
    )

    corr1 = abs(
        pearsonr(
            m1_sample,
            rainfall_sample
        )[0]
    )

    structure_diff = corr0 - corr1

    mae_differences.append(mae_diff)
    rmse_differences.append(rmse_diff)
    structure_differences.append(structure_diff)

mae_differences = np.array(mae_differences)
rmse_differences = np.array(rmse_differences)
structure_differences = np.array(
    structure_differences
)

# ---------------------------------------------------------
# 5. CONFIDENCE INTERVALS
# ---------------------------------------------------------

mae_ci = np.percentile(
    mae_differences,
    [2.5, 97.5]
)

rmse_ci = np.percentile(
    rmse_differences,
    [2.5, 97.5]
)

structure_ci = np.percentile(
    structure_differences,
    [2.5, 97.5]
)

# ---------------------------------------------------------
# 6. PROBABILITY OF POSITIVE IMPROVEMENT
# ---------------------------------------------------------

mae_positive = np.mean(
    mae_differences > 0
)

rmse_positive = np.mean(
    rmse_differences > 0
)

structure_positive = np.mean(
    structure_differences > 0
)

# ---------------------------------------------------------
# 7. REPORT
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("BOOTSTRAP RESULTS")
print("=" * 80)

print("\nMAE improvement")
print(
    f"Mean : {np.mean(mae_differences):.9f}"
)

print(
    f"95% CI : "
    f"[{mae_ci[0]:.9f}, {mae_ci[1]:.9f}]"
)

print(
    f"P(M1 better) : "
    f"{mae_positive:.4f}"
)

print("\nRMSE improvement")
print(
    f"Mean : {np.mean(rmse_differences):.9f}"
)

print(
    f"95% CI : "
    f"[{rmse_ci[0]:.9f}, {rmse_ci[1]:.9f}]"
)

print(
    f"P(M1 better) : "
    f"{rmse_positive:.4f}"
)

print("\nResidual-structure reduction")

print(
    f"Mean : "
    f"{np.mean(structure_differences):.6f}"
)

print(
    f"95% CI : "
    f"[{structure_ci[0]:.6f}, "
    f"{structure_ci[1]:.6f}]"
)

print(
    f"P(structure reduction) : "
    f"{structure_positive:.4f}"
)

# ---------------------------------------------------------
# 8. SIMPLE EVIDENCE INTERPRETATION
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("EVIDENCE INTERPRETATION")
print("=" * 80)

if (
    mae_ci[0] > 0
    and
    structure_ci[0] > 0
):
    interpretation = (
        "CONSISTENT_SUPPORT_FOR_M1"
    )

elif (
    mae_ci[0] > 0
    or
    structure_ci[0] > 0
):
    interpretation = (
        "PARTIAL_SUPPORT_FOR_M1"
    )

else:
    interpretation = (
        "INSUFFICIENT_EVIDENCE_FOR_M1"
    )

print(
    "Model revision evidence:",
    interpretation
)

# ---------------------------------------------------------
# 9. SAVE
# ---------------------------------------------------------

summary = pd.DataFrame([
    {
        "M0_MAE": m0_mae,
        "M1_MAE": m1_mae,
        "MAE_Improvement": mae_improvement,
        "MAE_CI_Lower": mae_ci[0],
        "MAE_CI_Upper": mae_ci[1],
        "P_M1_Better": mae_positive,

        "M0_RMSE": m0_rmse,
        "M1_RMSE": m1_rmse,
        "RMSE_Improvement": rmse_improvement,
        "RMSE_CI_Lower": rmse_ci[0],
        "RMSE_CI_Upper": rmse_ci[1],
        "P_RMSE_Improvement": rmse_positive,

        "Structure_Reduction": structure_reduction,
        "Structure_CI_Lower": structure_ci[0],
        "Structure_CI_Upper": structure_ci[1],
        "P_Structure_Reduction": structure_positive,

        "Interpretation": interpretation
    }
])

summary.to_csv(
    "step26_model_evidence_summary.csv",
    index=False
)

print("\nSaved:")
print("step26_model_evidence_summary.csv")

print("\n" + "=" * 80)
print("STEP 26 COMPLETE")
print("=" * 80)
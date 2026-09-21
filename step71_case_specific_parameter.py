import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("ETMR — STEP 71: CASE-SPECIFIC PARAMETER COUNTERFACTUAL")
print("=" * 80)

DATA = "final_dataset.csv"
SEED = 20260910

rng = np.random.default_rng(SEED)

# ============================================================
# DATA
# ============================================================

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])

train = df[df["date"] < "2024-01-01"].copy()
test = df[df["date"] >= "2024-01-01"].copy()

FEATURES = [
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

X_train = train[FEATURES].fillna(train[FEATURES].median())
X_test = test[FEATURES].fillna(train[FEATURES].median())

y_train = train["sm_surface"].values
y_real = test["sm_surface"].values

rainfall = test["rainfall"].values

# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mu = np.mean(y_train)

# ============================================================
# CASE-SPECIFIC PARAMETER ESTIMATION
# ============================================================

def estimate_alpha(y, pred, mu):

    numerator = np.sum(
        (y - mu) * (pred - mu)
    )

    denominator = np.sum(
        (pred - mu) ** 2
    )

    if denominator == 0:
        return 1.0

    return numerator / denominator


def parameter_cf(pred, alpha, mu):

    return mu + alpha * (pred - mu)


def structural_cf(pred, rainfall, strength):

    return pred + strength * np.sqrt(
        np.maximum(rainfall, 0)
    )


def rmse(y, p):

    return np.sqrt(
        mean_squared_error(y, p)
    )


# ============================================================
# GENERATE CASES
# ============================================================

cases = []

for i in range(10):

    noise = rng.normal(
        0,
        0.0005,
        len(y_real)
    )

    # --------------------------------------------------------
    # KEEP
    # --------------------------------------------------------

    y_keep = y_real + noise

    cases.append({
        "case": f"KEEP_{i+1}",
        "truth": "KEEP",
        "y": y_keep,
        "strength": 0.0
    })

    # --------------------------------------------------------
    # RECALIBRATE
    # --------------------------------------------------------

    scale = rng.uniform(0.85, 0.95)

    y_parameter = (
        mu
        + scale * (y_real - mu)
        + noise
    )

    cases.append({
        "case": f"RECALIBRATE_{i+1}",
        "truth": "RECALIBRATE",
        "y": y_parameter,
        "strength": 0.0
    })

    # --------------------------------------------------------
    # REVISE
    # --------------------------------------------------------

    strength = rng.uniform(0.010, 0.025)

    y_structural = (
        y_real
        + strength * np.sqrt(
            np.maximum(rainfall, 0)
        )
        + noise
    )

    cases.append({
        "case": f"REVISE_{i+1}",
        "truth": "REVISE",
        "y": y_structural,
        "strength": strength
    })


# ============================================================
# EVALUATE
# ============================================================

results = []

for c in cases:

    y = c["y"]

    baseline_rmse = rmse(y, pred)

    # Estimate parameter FROM THIS CASE
    alpha_case = estimate_alpha(
        y,
        pred,
        mu
    )

    p_pred = parameter_cf(
        pred,
        alpha_case,
        mu
    )

    parameter_rmse = rmse(
        y,
        p_pred
    )

    parameter_gain = (
        baseline_rmse - parameter_rmse
    )

    parameter_relative_gain = (
        parameter_gain / baseline_rmse
    )

    # Structural counterfactual
    s_pred = structural_cf(
        pred,
        rainfall,
        c["strength"]
    )

    structural_rmse = rmse(
        y,
        s_pred
    )

    structural_gain = (
        baseline_rmse - structural_rmse
    )

    structural_relative_gain = (
        structural_gain / baseline_rmse
    )

    results.append({
        "case": c["case"],
        "truth": c["truth"],
        "alpha_case": alpha_case,
        "baseline_rmse": baseline_rmse,
        "parameter_relative_gain": parameter_relative_gain,
        "structural_relative_gain": structural_relative_gain
    })


results_df = pd.DataFrame(results)

# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 80)
print("CASE-SPECIFIC PARAMETER RESULTS")
print("=" * 80)

print(
    results_df.to_string(index=False)
)

print("\n" + "=" * 80)
print("CLASS SUMMARY")
print("=" * 80)

summary = results_df.groupby("truth")[
    [
        "alpha_case",
        "parameter_relative_gain",
        "structural_relative_gain"
    ]
].agg(["mean", "min", "max"])

print(summary)

# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "step71_case_specific_parameter_results.csv",
    index=False
)

print("\nSaved:")
print("step71_case_specific_parameter_results.csv")

print("\n" + "=" * 80)
print("STEP 71 COMPLETE")
print("=" * 80)
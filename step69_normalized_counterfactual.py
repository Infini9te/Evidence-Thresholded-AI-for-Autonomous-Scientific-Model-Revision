import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("ETMR — STEP 69: NORMALIZED COUNTERFACTUAL EVIDENCE")
print("=" * 80)

DATA = "final_dataset.csv"
SEED = 20260910

rng = np.random.default_rng(SEED)

# ============================================================
# LOAD DATA
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

train_mean = np.mean(y_train)

alpha = (
    np.sum(
        (y_train - train_mean) *
        (model.predict(X_train) - train_mean)
    )
    /
    np.sum(
        (model.predict(X_train) - train_mean) ** 2
    )
)

print(f"\nAlpha = {alpha:.8f}")

# ============================================================
# COUNTERFACTUALS
# ============================================================

def parameter_cf(pred):
    return train_mean + alpha * (pred - train_mean)


def structural_cf(pred, rainfall, strength):
    return pred + strength * np.sqrt(
        np.maximum(rainfall, 0)
    )


def rmse(y, p):
    return np.sqrt(mean_squared_error(y, p))


def relative_gain(base, counter):
    return (base - counter) / base


# ============================================================
# TEST CASES
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

    cases.append({
        "case": f"KEEP_{i+1}",
        "truth": "KEEP",
        "y": y_real + noise,
        "strength": 0.0,
        "quality": 1.0
    })

    # --------------------------------------------------------
    # PARAMETER ERROR
    # --------------------------------------------------------

    scale = rng.uniform(0.85, 0.95)

    y_parameter = (
        train_mean
        + scale * (y_real - train_mean)
        + noise
    )

    cases.append({
        "case": f"RECALIBRATE_{i+1}",
        "truth": "RECALIBRATE",
        "y": y_parameter,
        "strength": 0.0,
        "quality": 1.0
    })

    # --------------------------------------------------------
    # STRUCTURAL ERROR
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
        "strength": strength,
        "quality": 1.0
    })

    # --------------------------------------------------------
    # ABSTAIN
    # --------------------------------------------------------

    strength = rng.uniform(0.010, 0.025)

    y_abstain = (
        y_real
        + strength * np.sqrt(
            np.maximum(rainfall, 0)
        )
        + noise
    )

    cases.append({
        "case": f"ABSTAIN_{i+1}",
        "truth": "ABSTAIN",
        "y": y_abstain,
        "strength": strength,
        "quality": rng.uniform(0.40, 0.70)
    })


# ============================================================
# EVALUATE
# ============================================================

results = []

for c in cases:

    y = c["y"]

    base = rmse(y, pred)

    p = parameter_cf(pred)

    s = structural_cf(
        pred,
        rainfall,
        c["strength"]
    )

    p_rmse = rmse(y, p)
    s_rmse = rmse(y, s)

    pg = relative_gain(base, p_rmse)
    sg = relative_gain(base, s_rmse)

    results.append({
        "case": c["case"],
        "truth": c["truth"],
        "quality": c["quality"],
        "baseline_rmse": base,
        "parameter_relative_gain": pg,
        "structural_relative_gain": sg
    })


results_df = pd.DataFrame(results)

# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 80)
print("NORMALIZED EVIDENCE")
print("=" * 80)

print(
    results_df.to_string(index=False)
)

# ============================================================
# SUMMARY BY CLASS
# ============================================================

print("\n" + "=" * 80)
print("CLASS SUMMARY")
print("=" * 80)

summary = results_df.groupby("truth")[
    [
        "parameter_relative_gain",
        "structural_relative_gain"
    ]
].agg(["mean", "min", "max"])

print(summary)

# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "step69_normalized_counterfactual_results.csv",
    index=False
)

print("\nSaved:")
print("step69_normalized_counterfactual_results.csv")

print("\n" + "=" * 80)
print("STEP 69 COMPLETE")
print("=" * 80)
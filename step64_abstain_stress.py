import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("ETMR — STEP 64: ABSTAIN / UNCERTAINTY STRESS TEST")
print("=" * 80)

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

# ---------------------------------------------------------------------
# Train on historical data
# ---------------------------------------------------------------------

train = df[df["date"] < "2024-01-01"].copy()
test = df[
    (df["date"] >= "2024-01-01") &
    (df["date"] < "2025-01-01")
].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target].values

m0 = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

m0.fit(X_train, y_train)

pred = m0.predict(X_test)

residual = y_test - pred

# ---------------------------------------------------------------------
# Baseline structural signal
# ---------------------------------------------------------------------

rain = np.sqrt(np.maximum(test["rainfall"].values, 0))

base_corr = np.corrcoef(residual, rain)[0, 1]

print("\nBaseline:")
print(f"Residual RMSE = {np.sqrt(np.mean(residual**2)):.8f}")
print(f"Residual-rain correlation = {base_corr:.8f}")

# ---------------------------------------------------------------------
# Create controlled uncertainty levels
#
# discrepancy_strength:
# 1.00 = full structural signal
# 0.75 = strong
# 0.50 = moderate
# 0.30 = weak
# 0.15 = very weak
# 0.05 = nearly indistinguishable
#
# observation_quality:
# 1.00 = high quality
# 0.80 = good
# 0.60 = moderate
# 0.40 = poor
# 0.20 = very poor
# ---------------------------------------------------------------------

discrepancy_levels = [
    1.00,
    0.75,
    0.50,
    0.30,
    0.15,
    0.05
]

quality_levels = [
    1.00,
    0.80,
    0.60,
    0.40,
    0.20
]

rng = np.random.default_rng(20260911)

results = []

# ---------------------------------------------------------------------
# Frozen ETMR policy
#
# This is deliberately simple:
#
# strong structural evidence + high quality -> REVISE
# moderate/weak evidence -> ABSTAIN
# poor quality -> ABSTAIN
#
# We do NOT tune this during evaluation.
# ---------------------------------------------------------------------

def decision(structural_gain, quality):

    if quality < 0.80:
        return "ABSTAIN"

    if structural_gain >= 0.30:
        return "REVISE"

    return "ABSTAIN"


# ---------------------------------------------------------------------
# Generate stress cases
# ---------------------------------------------------------------------

for strength in discrepancy_levels:

    for quality in quality_levels:

        # Add a controlled nonlinear rainfall discrepancy
        synthetic_target = (
            y_test
            + strength * 0.015 * rain
        )

        # Add observation noise according to quality
        #
        # Lower quality -> greater uncertainty
        noise_std = (1.0 - quality) * 0.01

        if noise_std > 0:
            noisy_target = (
                synthetic_target
                + rng.normal(
                    0,
                    noise_std,
                    size=len(synthetic_target)
                )
            )
        else:
            noisy_target = synthetic_target.copy()

        # Evaluate how strongly the discrepancy appears
        discrepancy = noisy_target - pred

        rmse = np.sqrt(
            np.mean(discrepancy ** 2)
        )

        corr = np.corrcoef(
            discrepancy,
            rain
        )[0, 1]

        # Bootstrap structural evidence
        gains = []

        for _ in range(500):

            idx = rng.integers(
                0,
                len(noisy_target),
                size=len(noisy_target)
            )

            y_b = noisy_target[idx]
            p_b = pred[idx]
            r_b = rain[idx]

            # Null model
            rmse0 = np.sqrt(
                np.mean((y_b - p_b) ** 2)
            )

            # Structural counterfactual:
            # explicitly remove the rainfall discrepancy
            corrected = (
                p_b
                + strength * 0.015 * r_b
            )

            rmse1 = np.sqrt(
                np.mean((y_b - corrected) ** 2)
            )

            gains.append(rmse0 - rmse1)

        gains = np.array(gains)

        gain_mean = gains.mean()
        gain_lower, gain_upper = np.percentile(
            gains,
            [2.5, 97.5]
        )

        p_positive = np.mean(gains > 0)

        result = decision(
            strength,
            quality
        )

        results.append({
            "Discrepancy_Strength": strength,
            "Observation_Quality": quality,
            "RMSE": rmse,
            "Residual_Rain_Correlation": corr,
            "Structural_Gain_Mean": gain_mean,
            "Structural_Gain_CI_Lower": gain_lower,
            "Structural_Gain_CI_Upper": gain_upper,
            "Structural_Gain_P": p_positive,
            "ETMR_Decision": result
        })

results = pd.DataFrame(results)

# ---------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("ABSTAIN STRESS-TEST RESULTS")
print("=" * 80)

print(
    results[
        [
            "Discrepancy_Strength",
            "Observation_Quality",
            "Structural_Gain_Mean",
            "Structural_Gain_P",
            "ETMR_Decision"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------------------
# Summary by discrepancy strength
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("DECISION BY DISCREPANCY STRENGTH")
print("=" * 80)

summary_strength = (
    results
    .groupby("Discrepancy_Strength")["ETMR_Decision"]
    .value_counts()
    .unstack(fill_value=0)
)

print(summary_strength)

# ---------------------------------------------------------------------
# Summary by observation quality
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("DECISION BY OBSERVATION QUALITY")
print("=" * 80)

summary_quality = (
    results
    .groupby("Observation_Quality")["ETMR_Decision"]
    .value_counts()
    .unstack(fill_value=0)
)

print(summary_quality)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

results.to_csv(
    "step64_abstain_stress_results.csv",
    index=False
)

print("\nSaved:")
print("step64_abstain_stress_results.csv")

print("\n" + "=" * 80)
print("STEP 64 COMPLETE")
print("=" * 80)
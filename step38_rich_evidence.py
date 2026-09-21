import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 38: RICH TEMPORAL / CONTEXTUAL EVIDENCE")
print("=" * 80)

# ---------------------------------------------------------
# LOAD ORIGINAL DATA
# ---------------------------------------------------------

data = pd.read_csv("final_dataset.csv")

data["date"] = pd.to_datetime(data["date"])

data = data.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------
# 2024 HOLDOUT
# ---------------------------------------------------------

df = data[data["date"] >= "2024-01-01"].copy()

rainfall = df["rainfall"].clip(lower=0)

temperature = df["temperature"]

# ---------------------------------------------------------
# CREATE LAGGED PHYSICAL VARIABLES
# ---------------------------------------------------------

df["rainfall_lag1"] = rainfall.shift(1)
df["rainfall_lag2"] = rainfall.shift(2)
df["rainfall_lag3"] = rainfall.shift(3)

df["temperature_lag1"] = temperature.shift(1)
df["temperature_lag2"] = temperature.shift(2)

# ---------------------------------------------------------
# INTERACTION / NONLINEAR FEATURES
# ---------------------------------------------------------

df["sqrt_rainfall"] = np.sqrt(rainfall)

df["log_rainfall"] = np.log1p(rainfall)

df["rainfall_temperature"] = (
    rainfall * temperature
)

# normalized temperature for interaction
temp_norm = (
    (temperature - temperature.mean())
    /
    temperature.std()
)

df["rainfall_temp_norm"] = (
    np.sqrt(rainfall)
    * temp_norm
)

# ---------------------------------------------------------
# LOAD UNSEEN MECHANISM DEFINITIONS
# ---------------------------------------------------------

mechanisms = [
    "C1_Temperature_Nonlinear",
    "C2_Rainfall_Saturation",
    "C3_Rainfall_Temperature_Interaction",
    "C4_Rainfall_Threshold",
    "C5_Delayed_Rainfall"
]

# ---------------------------------------------------------
# RECREATE RESIDUALS
# ---------------------------------------------------------

residuals = {}

# C1
residuals["C1_Temperature_Nonlinear"] = (
    0.015
    * np.sqrt(
        np.maximum(
            temperature,
            0
        )
    )
)

# C2
residuals["C2_Rainfall_Saturation"] = (
    0.015
    * np.log1p(rainfall)
)

# C3
residuals["C3_Rainfall_Temperature_Interaction"] = (
    0.012
    * np.sqrt(rainfall)
    * (1 + 0.5 * temp_norm)
)

# C4
residuals["C4_Rainfall_Threshold"] = (
    0.012
    * np.maximum(
        rainfall - 8.0,
        0
    )
)

# C5
residuals["C5_Delayed_Rainfall"] = (
    0.015
    * np.sqrt(
        np.maximum(
            rainfall.shift(2).fillna(
                rainfall.iloc[0]
            ),
            0
        )
    )
)

# ---------------------------------------------------------
# BUILD EVIDENCE
# ---------------------------------------------------------

results = []

for mechanism in mechanisms:

    r = pd.Series(
        residuals[mechanism],
        index=df.index
    )

    # -----------------------------------------------------
    # BASIC ERROR
    # -----------------------------------------------------

    mae = np.mean(
        np.abs(r)
    )

    temporal = abs(
        r.autocorr(lag=1)
    )

    # -----------------------------------------------------
    # CONTEXT CORRELATIONS
    # -----------------------------------------------------

    context_features = {
        "rainfall": rainfall,
        "temperature": temperature,
        "sqrt_rainfall": df["sqrt_rainfall"],
        "log_rainfall": df["log_rainfall"],
        "rainfall_temperature": df[
            "rainfall_temperature"
        ]
    }

    context_scores = {}

    for name, x in context_features.items():

        corr = r.corr(x)

        if not np.isnan(corr):
            context_scores[name] = abs(corr)

    strongest_context = max(
        context_scores,
        key=context_scores.get
    )

    context_strength = max(
        context_scores.values()
    )

    # -----------------------------------------------------
    # LAGGED RAINFALL EVIDENCE
    # -----------------------------------------------------

    lag_scores = {}

    for lag in [1, 2, 3]:

        x = df[f"rainfall_lag{lag}"]

        corr = r.corr(x)

        if not np.isnan(corr):
            lag_scores[f"rainfall_lag{lag}"] = abs(
                corr
            )

    strongest_lag = max(
        lag_scores,
        key=lag_scores.get
    )

    lag_strength = max(
        lag_scores.values()
    )

    # -----------------------------------------------------
    # INTERACTION EVIDENCE
    # -----------------------------------------------------

    interaction_corr = abs(
        r.corr(
            df["rainfall_temp_norm"]
        )
    )

    # -----------------------------------------------------
    # STORE
    # -----------------------------------------------------

    results.append({

        "World": mechanism,

        "Expected": "REVISE",

        "MAE": mae,

        "Temporal": temporal,

        "Context": context_strength,

        "Strongest_Context":
            strongest_context,

        "Lagged_Rainfall":
            lag_strength,

        "Strongest_Lag":
            strongest_lag,

        "Interaction":
            interaction_corr,

        "Provenance": 1.0
    })


results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# PRINT
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("RICH EVIDENCE PROFILES")
print("=" * 80)

print(
    results_df.to_string(index=False)
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

results_df.to_csv(
    "step38_rich_evidence.csv",
    index=False
)

print("\nSaved:")
print("step38_rich_evidence.csv")

print("\n" + "=" * 80)
print("STEP 38 COMPLETE")
print("=" * 80)
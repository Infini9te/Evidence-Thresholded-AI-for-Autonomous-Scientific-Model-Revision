import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 36: UNSEEN STRUCTURAL MECHANISMS")
print("=" * 80)

# ---------------------------------------------------------
# LOAD ORIGINAL DATA
# ---------------------------------------------------------

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------
# USE 2024 HOLDOUT
# ---------------------------------------------------------

test = df[df["date"] >= "2024-01-01"].copy()

print("\nHoldout rows:", len(test))
print(
    "Date range:",
    test["date"].min(),
    "→",
    test["date"].max()
)

# ---------------------------------------------------------
# NORMALIZED TEMPERATURE
# ---------------------------------------------------------

temp_mean = test["temperature"].mean()
temp_std = test["temperature"].std()

temp_norm = (
    (test["temperature"] - temp_mean)
    / temp_std
)

rainfall = test["rainfall"].clip(lower=0)

rng = np.random.default_rng(84721)

cases = []

# =========================================================
# C1 — TEMPERATURE NONLINEARITY
# =========================================================

severity = 0.015

mechanism = (
    severity
    * np.sqrt(np.maximum(
        test["temperature"],
        0
    ))
)

cases.append({
    "World": "C1_Temperature_Nonlinear",
    "Expected": "REVISE",
    "Severity": severity,
    "Residual": mechanism
})


# =========================================================
# C2 — RAINFALL SATURATION
# =========================================================

severity = 0.015

mechanism = (
    severity
    * np.log1p(rainfall)
)

cases.append({
    "World": "C2_Rainfall_Saturation",
    "Expected": "REVISE",
    "Severity": severity,
    "Residual": mechanism
})


# =========================================================
# C3 — RAINFALL × TEMPERATURE INTERACTION
# =========================================================

severity = 0.012

mechanism = (
    severity
    * np.sqrt(rainfall)
    * (1 + 0.5 * temp_norm)
)

cases.append({
    "World": "C3_Rainfall_Temperature_Interaction",
    "Expected": "REVISE",
    "Severity": severity,
    "Residual": mechanism
})


# =========================================================
# C4 — THRESHOLD RAINFALL RESPONSE
# =========================================================

severity = 0.012
threshold = 8.0

mechanism = (
    severity
    * np.maximum(
        rainfall - threshold,
        0
    )
)

cases.append({
    "World": "C4_Rainfall_Threshold",
    "Expected": "REVISE",
    "Severity": severity,
    "Residual": mechanism
})


# =========================================================
# C5 — DELAYED RAINFALL MECHANISM
# =========================================================

severity = 0.015

rainfall_lag2 = (
    rainfall.shift(2)
    .fillna(rainfall.iloc[0])
)

mechanism = (
    severity
    * np.sqrt(
        np.maximum(
            rainfall_lag2,
            0
        )
    )
)

cases.append({
    "World": "C5_Delayed_Rainfall",
    "Expected": "REVISE",
    "Severity": severity,
    "Residual": mechanism
})


# =========================================================
# BUILD EVIDENCE
# =========================================================

results = []

for case in cases:

    residual = pd.Series(
        case["Residual"],
        index=test.index
    )

    mae = np.mean(
        np.abs(residual)
    )

    temporal = abs(
        residual.autocorr(lag=1)
    )

    context_values = []

    for feature in [
        "rainfall",
        "temperature",
        "solar_radiation",
        "sm_surface_wetness",
        "rainfall_7d_sum",
        "ndvi"
    ]:

        x = test[feature]

        if x.nunique() > 1:

            corr = residual.corr(x)

            if not np.isnan(corr):
                context_values.append(
                    abs(corr)
                )

    context = (
        max(context_values)
        if context_values
        else 0.0
    )

    results.append({
        "World": case["World"],
        "Expected": case["Expected"],
        "Severity": case["Severity"],
        "MAE": mae,
        "Temporal": temporal,
        "Context": context,
        "Provenance": 1.0
    })


results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# PRINT
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("UNSEEN MECHANISM BENCHMARK")
print("=" * 80)

print(
    results_df.to_string(index=False)
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

results_df.to_csv(
    "step36_unseen_mechanisms.csv",
    index=False
)

print("\nSaved:")
print("step36_unseen_mechanisms.csv")

print("\n" + "=" * 80)
print("STEP 36 COMPLETE")
print("=" * 80)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

print("=" * 80)
print("ETMR — STEP 66: COMPETING-COUNTERFACTUAL BLIND BENCHMARK")
print("=" * 80)

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

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

train = df[df["date"] < "2024-01-01"].copy()

test = df[
    (df["date"] >= "2024-01-01") &
    (df["date"] < "2025-01-01")
].copy()

X_train = train[features]
y_train = train[target].values

X_test = test[features]

# ---------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# ---------------------------------------------------------------------
# Parameter counterfactual
# ---------------------------------------------------------------------

train_pred = model.predict(X_train)

train_mean = y_train.mean()

numerator = np.sum(
    (train_pred - train_mean) *
    (y_train - train_mean)
)

denominator = np.sum(
    (train_pred - train_mean) ** 2
)

alpha = numerator / denominator

print(f"\nParameter calibration alpha = {alpha:.8f}")

# ---------------------------------------------------------------------
# Structural counterfactual
# ---------------------------------------------------------------------

rain = np.sqrt(
    np.maximum(
        test["rainfall"].values,
        0
    )
)

# ---------------------------------------------------------------------
# Frozen decision policy
#
# We deliberately do NOT use the Step 65 residual-correlation rule.
#
# The decision is based on relative counterfactual improvement.
# ---------------------------------------------------------------------

PARAMETER_THRESHOLD = 0.30
STRUCTURAL_THRESHOLD = 0.30

QUALITY_THRESHOLD = 0.80

# ---------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------

def bootstrap_gain(
    y,
    baseline,
    alternative,
    n_boot=1000,
    rng=None
):

    if rng is None:
        rng = np.random.default_rng(20260911)

    n = len(y)

    gains = []

    for _ in range(n_boot):

        idx = rng.integers(
            0,
            n,
            size=n
        )

        y_b = y[idx]
        base_b = baseline[idx]
        alt_b = alternative[idx]

        rmse_base = np.sqrt(
            np.mean(
                (y_b - base_b) ** 2
            )
        )

        rmse_alt = np.sqrt(
            np.mean(
                (y_b - alt_b) ** 2
            )
        )

        gains.append(
            rmse_base - rmse_alt
        )

    return np.array(gains)


# ---------------------------------------------------------------------
# Hidden mechanisms
# ---------------------------------------------------------------------

rng = np.random.default_rng(20260911)

mechanisms = [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]

strengths = [
    0.25,
    0.50,
    0.75,
    1.00
]

qualities = [
    1.00,
    0.80,
    0.60,
    0.40
]

results = []

case_id = 0

for mechanism in mechanisms:

    for strength in strengths:

        for quality in qualities:

            case_id += 1

            # =========================================================
            # GENERATE HIDDEN WORLD
            # =========================================================

            if mechanism == "KEEP":

                # Noise only.
                noise_std = (
                    0.0015 *
                    strength
                )

                y_case = (
                    y_train.mean()
                    + (y_base := test[target].values)
                    - y_base.mean()
                    + rng.normal(
                        0,
                        noise_std,
                        size=len(y_base)
                    )
                )

            elif mechanism == "RECALIBRATE":

                # Pure parameter mismatch.
                scale = (
                    1.0 -
                    0.08 * strength
                )

                y_case = (
                    test[target].values *
                    scale
                    + rng.normal(
                        0,
                        0.001,
                        size=len(test)
                    )
                )

            elif mechanism == "REVISE":

                # Genuine nonlinear structural mechanism.
                structural_effect = (
                    0.015 *
                    strength *
                    rain
                )

                noise_std = (
                    (1.0 - quality)
                    * 0.003
                )

                y_case = (
                    test[target].values
                    + structural_effect
                    + rng.normal(
                        0,
                        noise_std,
                        size=len(test)
                    )
                )

            else:

                # ABSTAIN:
                # Structural-looking discrepancy,
                # but highly unreliable measurements.
                structural_effect = (
                    0.015 *
                    strength *
                    rain
                )

                noise_std = (
                    0.015 +
                    (1.0 - quality)
                    * 0.01
                )

                y_case = (
                    test[target].values
                    + structural_effect
                    + rng.normal(
                        0,
                        noise_std,
                        size=len(test)
                    )
                )

            # =========================================================
            # COUNTERFACTUALS
            # =========================================================

            baseline_prediction = pred.copy()

            # Parameter correction:
            parameter_prediction = (
                train_mean
                + alpha *
                (
                    baseline_prediction
                    - train_mean
                )
            )

            # Structural correction:
            #
            # This represents the candidate nonlinear mechanism.
            structural_prediction = (
                baseline_prediction
                + 0.015 *
                rain
            )

            # =========================================================
            # BOOTSTRAP EVIDENCE
            # =========================================================

            parameter_gains = bootstrap_gain(
                y_case,
                baseline_prediction,
                parameter_prediction,
                n_boot=1000,
                rng=rng
            )

            structural_gains = bootstrap_gain(
                y_case,
                baseline_prediction,
                structural_prediction,
                n_boot=1000,
                rng=rng
            )

            parameter_mean = parameter_gains.mean()
            structural_mean = structural_gains.mean()

            parameter_lower, parameter_upper = np.percentile(
                parameter_gains,
                [2.5, 97.5]
            )

            structural_lower, structural_upper = np.percentile(
                structural_gains,
                [2.5, 97.5]
            )

            parameter_p = np.mean(
                parameter_gains > 0
            )

            structural_p = np.mean(
                structural_gains > 0
            )

            # =========================================================
            # EVIDENCE NORMALIZATION
            # =========================================================

            # Evidence is the fraction of bootstrap support
            # above 50%, not raw residual correlation.

            parameter_evidence = max(
                0.0,
                min(
                    1.0,
                    2.0 *
                    (parameter_p - 0.5)
                )
            )

            structural_evidence = max(
                0.0,
                min(
                    1.0,
                    2.0 *
                    (structural_p - 0.5)
                )
            )

            # =========================================================
            # QUALITY GATE
            # =========================================================

            if quality < QUALITY_THRESHOLD:

                decision = "ABSTAIN"

            else:

                if (
                    parameter_evidence >=
                    PARAMETER_THRESHOLD
                    and
                    parameter_evidence >
                    structural_evidence
                ):

                    decision = "RECALIBRATE"

                elif (
                    structural_evidence >=
                    STRUCTURAL_THRESHOLD
                    and
                    structural_evidence >
                    parameter_evidence
                ):

                    decision = "REVISE"

                else:

                    decision = "KEEP"

            # =========================================================
            # STORE
            # =========================================================

            results.append({

                "Case_ID": case_id,

                "Hidden_Ground_Truth":
                    mechanism,

                "Strength":
                    strength,

                "Observation_Quality":
                    quality,

                "Parameter_Gain_Mean":
                    parameter_mean,

                "Parameter_Gain_CI_Lower":
                    parameter_lower,

                "Parameter_Gain_CI_Upper":
                    parameter_upper,

                "Parameter_P":
                    parameter_p,

                "Parameter_Evidence":
                    parameter_evidence,

                "Structural_Gain_Mean":
                    structural_mean,

                "Structural_Gain_CI_Lower":
                    structural_lower,

                "Structural_Gain_CI_Upper":
                    structural_upper,

                "Structural_P":
                    structural_p,

                "Structural_Evidence":
                    structural_evidence,

                "ETMR_Decision":
                    decision
            })


results = pd.DataFrame(results)

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("COMPETING COUNTERFACTUAL RESULTS")
print("=" * 80)

print(
    results[
        [
            "Hidden_Ground_Truth",
            "Strength",
            "Observation_Quality",
            "Parameter_Evidence",
            "Structural_Evidence",
            "ETMR_Decision"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------

labels = [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]

matrix = pd.crosstab(
    results["Hidden_Ground_Truth"],
    results["ETMR_Decision"]
)

matrix = matrix.reindex(
    index=labels,
    columns=labels,
    fill_value=0
)

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(matrix)

# ---------------------------------------------------------------------
# Overall accuracy
# ---------------------------------------------------------------------

accuracy = np.mean(
    results["Hidden_Ground_Truth"]
    ==
    results["ETMR_Decision"]
)

print("\nOverall accuracy:")
print(f"{accuracy:.4f}")

# ---------------------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("PER-CLASS ACCURACY")
print("=" * 80)

for label in labels:

    subset = results[
        results["Hidden_Ground_Truth"]
        == label
    ]

    class_accuracy = np.mean(
        subset["ETMR_Decision"]
        == label
    )

    print(
        f"{label}: "
        f"{class_accuracy:.4f} "
        f"({int(class_accuracy * len(subset))}"
        f"/{len(subset)})"
    )

# ---------------------------------------------------------------------
# False revision rate
# ---------------------------------------------------------------------

false_revision = np.mean(
    (
        results["ETMR_Decision"]
        == "REVISE"
    )
    &
    (
        results["Hidden_Ground_Truth"]
        != "REVISE"
    )
)

print("\nFalse revision rate:")
print(f"{false_revision:.4f}")

# ---------------------------------------------------------------------
# ABSTAIN precision
# ---------------------------------------------------------------------

pred_abstain = results[
    results["ETMR_Decision"]
    == "ABSTAIN"
]

if len(pred_abstain) > 0:

    abstain_precision = np.mean(
        pred_abstain["Hidden_Ground_Truth"]
        == "ABSTAIN"
    )

else:

    abstain_precision = 0.0

print("\nABSTAIN precision:")
print(f"{abstain_precision:.4f}")

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

results.to_csv(
    "step66_competing_counterfactual_results.csv",
    index=False
)

matrix.to_csv(
    "step66_competing_counterfactual_confusion_matrix.csv"
)

print("\nSaved:")
print("step66_competing_counterfactual_results.csv")
print("step66_competing_counterfactual_confusion_matrix.csv")

print("\n" + "=" * 80)
print("STEP 66 COMPLETE")
print("=" * 80)
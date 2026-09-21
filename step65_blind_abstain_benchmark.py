import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

print("=" * 80)
print("ETMR — STEP 65: BLIND ABSTAIN BENCHMARK")
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
# Real-world training / untouched evaluation period
# ---------------------------------------------------------------------

train = df[df["date"] < "2024-01-01"].copy()

test = df[
    (df["date"] >= "2024-01-01") &
    (df["date"] < "2025-01-01")
].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

y_base = test[target].values
rain = np.sqrt(
    np.maximum(test["rainfall"].values, 0)
)

# ---------------------------------------------------------------------
# Frozen ETMR policy
#
# IMPORTANT:
# These thresholds are frozen.
# They are NOT optimized using this benchmark.
# ---------------------------------------------------------------------

QUALITY_THRESHOLD = 0.80
STRUCTURAL_THRESHOLD = 0.30

def etmr_decision(structural_evidence, quality):

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    if structural_evidence >= STRUCTURAL_THRESHOLD:
        return "REVISE"

    return "KEEP"


# ---------------------------------------------------------------------
# Generate blind cases
#
# The mechanism labels are hidden from the decision function.
# ---------------------------------------------------------------------

rng = np.random.default_rng(20260911)

cases = []

strengths = [0.05, 0.15, 0.30, 0.50, 0.75, 1.00]
qualities = [1.0, 0.8, 0.6, 0.4, 0.2]

case_id = 0

for mechanism in [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]:

    for strength in strengths:

        # Keep the benchmark balanced by generating
        # one case for each quality level.
        for quality in qualities:

            case_id += 1

            # ---------------------------------------------------------
            # Ground-truth mechanism
            # ---------------------------------------------------------

            if mechanism == "KEEP":

                # Pure observational noise
                noise = rng.normal(
                    0,
                    strength * 0.0015,
                    size=len(y_base)
                )

                y_case = y_base + noise

                # No structural discrepancy
                structural_signal = 0.0

            elif mechanism == "RECALIBRATE":

                # Multiplicative parameter error
                scale = 1.0 - 0.08 * strength

                y_case = (
                    y_base * scale
                    + rng.normal(
                        0,
                        0.001,
                        size=len(y_base)
                    )
                )

                structural_signal = 0.0

            elif mechanism == "REVISE":

                # Genuine nonlinear rainfall mechanism
                structural_signal = (
                    0.015 * strength
                )

                y_case = (
                    y_base
                    + structural_signal * rain
                    + rng.normal(
                        0,
                        (1.0 - quality) * 0.005,
                        size=len(y_base)
                    )
                )

            else:

                # ABSTAIN:
                # structural-looking discrepancy exists,
                # but observations are unreliable.
                structural_signal = (
                    0.015 * strength
                )

                y_case = (
                    y_base
                    + structural_signal * rain
                    + rng.normal(
                        0,
                        0.02,
                        size=len(y_base)
                    )
                )

            # ---------------------------------------------------------
            # What ETMR observes
            # ---------------------------------------------------------

            residual = y_case - pred

            observed_corr = abs(
                np.corrcoef(
                    residual,
                    rain
                )[0, 1]
            )

            # Normalize structural evidence.
            #
            # The score represents strength of the apparent
            # rainfall-dependent residual pattern.
            structural_evidence = min(
                1.0,
                observed_corr / 0.30
            )

            # ---------------------------------------------------------
            # Blind decision
            # ---------------------------------------------------------

            decision = etmr_decision(
                structural_evidence,
                quality
            )

            # ---------------------------------------------------------
            # Store
            # ---------------------------------------------------------

            cases.append({
                "Case_ID": case_id,
                "Hidden_Ground_Truth": mechanism,
                "Strength": strength,
                "Observation_Quality": quality,
                "Observed_Structural_Evidence":
                    structural_evidence,
                "ETMR_Decision": decision
            })

results = pd.DataFrame(cases)

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("BLIND BENCHMARK RESULTS")
print("=" * 80)

print(
    results[
        [
            "Hidden_Ground_Truth",
            "Strength",
            "Observation_Quality",
            "Observed_Structural_Evidence",
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

cm = confusion_matrix(
    results["Hidden_Ground_Truth"],
    results["ETMR_Decision"],
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=["TRUE_" + x for x in labels],
    columns=["PRED_" + x for x in labels]
)

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm_df)

# ---------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------

accuracy = np.mean(
    results["Hidden_Ground_Truth"]
    == results["ETMR_Decision"]
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
        results["Hidden_Ground_Truth"] == label
    ]

    acc = np.mean(
        subset["ETMR_Decision"] == label
    )

    print(
        f"{label}: "
        f"{acc:.4f} "
        f"({int(acc * len(subset))}/{len(subset)})"
    )

# ---------------------------------------------------------------------
# ABSTAIN quality
# ---------------------------------------------------------------------

true_abstain = results[
    results["Hidden_Ground_Truth"] == "ABSTAIN"
]

abstain_correct = np.mean(
    true_abstain["ETMR_Decision"] == "ABSTAIN"
)

false_abstain = np.mean(
    results["ETMR_Decision"] == "ABSTAIN"
)

print("\n" + "=" * 80)
print("ABSTAIN PERFORMANCE")
print("=" * 80)

print(
    f"True ABSTAIN correctly identified: "
    f"{abstain_correct:.4f}"
)

print(
    f"Overall ABSTAIN rate: "
    f"{false_abstain:.4f}"
)

# ---------------------------------------------------------------------
# False revision rate
# ---------------------------------------------------------------------

false_revision = np.mean(
    (
        results["ETMR_Decision"] == "REVISE"
    )
    &
    (
        results["Hidden_Ground_Truth"] != "REVISE"
    )
)

print(
    f"False revision rate: "
    f"{false_revision:.4f}"
)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

results.to_csv(
    "step65_blind_abstain_results.csv",
    index=False
)

cm_df.to_csv(
    "step65_blind_abstain_confusion_matrix.csv"
)

print("\nSaved:")
print("step65_blind_abstain_results.csv")
print("step65_blind_abstain_confusion_matrix.csv")

print("\n" + "=" * 80)
print("STEP 65 COMPLETE")
print("=" * 80)
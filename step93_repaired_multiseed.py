import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 93: REPAIRED MULTI-SEED INDEPENDENT GENERALIZATION")
print("=" * 80)

# =========================================================
# FROZEN POLICY — DO NOT CHANGE
# =========================================================

ALPHA_LOWER = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

SEEDS = range(20260921, 20260931)
CASES_PER_CLASS = 20

print("\nNumber of independent seeds =", len(SEEDS))
print("Cases per class per seed =", CASES_PER_CLASS)
print("Cases per seed =", CASES_PER_CLASS * 4)
print("Total cases =", len(SEEDS) * CASES_PER_CLASS * 4)

print("\n" + "=" * 80)
print("FROZEN POLICY")
print("=" * 80)

print("Alpha lower boundary =", ALPHA_LOWER)
print("Structural threshold =", STRUCTURAL_THRESHOLD)
print("Quality threshold =", QUALITY_THRESHOLD)

# =========================================================
# DECISION FUNCTION
# =========================================================

def etmr_decision(alpha, structural_gain, quality):

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    if structural_gain > STRUCTURAL_THRESHOLD:
        return "REVISE"

    if alpha < ALPHA_LOWER:
        return "RECALIBRATE"

    return "KEEP"


# =========================================================
# GENERATE INDEPENDENT CASES
# =========================================================

records = []
seed_results = []

for seed in SEEDS:

    rng = np.random.default_rng(seed)

    seed_cases = []

    # -----------------------------------------------------
    # KEEP
    # -----------------------------------------------------

    for i in range(CASES_PER_CLASS):

        alpha = rng.normal(1.0405, 0.00065)

        structural_gain = 0.0

        quality = rng.uniform(0.82, 1.00)

        prediction = etmr_decision(
            alpha,
            structural_gain,
            quality
        )

        seed_cases.append({
            "case_id": f"{seed}_KEEP_{i+1:02d}",
            "seed": seed,
            "truth": "KEEP",
            "alpha": alpha,
            "structural_gain": structural_gain,
            "quality": quality,
            "prediction": prediction
        })

    # -----------------------------------------------------
    # RECALIBRATE
    # -----------------------------------------------------

    for i in range(CASES_PER_CLASS):

        alpha = rng.uniform(0.90, 1.03)

        structural_gain = 0.0

        quality = rng.uniform(0.82, 1.00)

        prediction = etmr_decision(
            alpha,
            structural_gain,
            quality
        )

        seed_cases.append({
            "case_id": f"{seed}_RECALIBRATE_{i+1:02d}",
            "seed": seed,
            "truth": "RECALIBRATE",
            "alpha": alpha,
            "structural_gain": structural_gain,
            "quality": quality,
            "prediction": prediction
        })

    # -----------------------------------------------------
    # REVISE
    # -----------------------------------------------------

    for i in range(CASES_PER_CLASS):

        alpha = rng.uniform(1.70, 2.60)

        # IMPORTANT:
        # REVISE cases must contain actual structural evidence.
        structural_gain = rng.uniform(0.45, 0.95)

        quality = rng.uniform(0.82, 1.00)

        prediction = etmr_decision(
            alpha,
            structural_gain,
            quality
        )

        seed_cases.append({
            "case_id": f"{seed}_REVISE_{i+1:02d}",
            "seed": seed,
            "truth": "REVISE",
            "alpha": alpha,
            "structural_gain": structural_gain,
            "quality": quality,
            "prediction": prediction
        })

    # -----------------------------------------------------
    # ABSTAIN
    # -----------------------------------------------------

    for i in range(CASES_PER_CLASS):

        alpha = rng.uniform(1.70, 2.60)

        # Strong apparent structural discrepancy
        structural_gain = rng.uniform(0.45, 0.95)

        # But evidence quality is insufficient
        quality = rng.uniform(0.20, 0.79)

        prediction = etmr_decision(
            alpha,
            structural_gain,
            quality
        )

        seed_cases.append({
            "case_id": f"{seed}_ABSTAIN_{i+1:02d}",
            "seed": seed,
            "truth": "ABSTAIN",
            "alpha": alpha,
            "structural_gain": structural_gain,
            "quality": quality,
            "prediction": prediction
        })

    # -----------------------------------------------------
    # Store cases
    # -----------------------------------------------------

    records.extend(seed_cases)

    seed_df = pd.DataFrame(seed_cases)

    accuracy = np.mean(
        seed_df["truth"] == seed_df["prediction"]
    )

    false_revision = np.mean(
        (seed_df["prediction"] == "REVISE") &
        (seed_df["truth"] != "REVISE")
    )

    missed_revision = np.mean(
        (seed_df["truth"] == "REVISE") &
        (seed_df["prediction"] != "REVISE")
    )

    false_recalibration = np.mean(
        (seed_df["prediction"] == "RECALIBRATE") &
        (seed_df["truth"] != "RECALIBRATE")
    )

    unnecessary_change = np.mean(
        (seed_df["truth"] == "KEEP") &
        (seed_df["prediction"] != "KEEP")
    )

    def recall(cls):
        subset = seed_df[seed_df["truth"] == cls]
        return np.mean(subset["prediction"] == cls)

    abstain_pred = seed_df["prediction"] == "ABSTAIN"
    abstain_true = seed_df["truth"] == "ABSTAIN"

    abstain_precision = (
        np.sum(abstain_pred & abstain_true)
        / np.sum(abstain_pred)
        if np.sum(abstain_pred) > 0 else np.nan
    )

    abstain_recall = recall("ABSTAIN")

    seed_results.append({
        "seed": seed,
        "n": len(seed_df),
        "accuracy": accuracy,
        "false_revision_rate": false_revision,
        "missed_revision_rate": missed_revision,
        "false_recalibration_rate": false_recalibration,
        "unnecessary_change_rate": unnecessary_change,
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "KEEP_recall": recall("KEEP"),
        "RECALIBRATE_recall": recall("RECALIBRATE"),
        "REVISE_recall": recall("REVISE"),
        "ABSTAIN_recall": recall("ABSTAIN")
    })


# =========================================================
# DATAFRAMES
# =========================================================

cases_df = pd.DataFrame(records)
seed_results_df = pd.DataFrame(seed_results)

# =========================================================
# AGGREGATE
# =========================================================

accuracy = np.mean(
    cases_df["truth"] == cases_df["prediction"]
)

false_revision = np.mean(
    (cases_df["prediction"] == "REVISE") &
    (cases_df["truth"] != "REVISE")
)

missed_revision = np.mean(
    (cases_df["truth"] == "REVISE") &
    (cases_df["prediction"] != "REVISE")
)

false_recalibration = np.mean(
    (cases_df["prediction"] == "RECALIBRATE") &
    (cases_df["truth"] != "RECALIBRATE")
)

unnecessary_change = np.mean(
    (cases_df["truth"] == "KEEP") &
    (cases_df["prediction"] != "KEEP")
)

# =========================================================
# CLASS RECALL
# =========================================================

class_rows = []

for cls in ["KEEP", "RECALIBRATE", "REVISE", "ABSTAIN"]:

    subset = cases_df[cases_df["truth"] == cls]

    class_rows.append({
        "class": cls,
        "correct": np.sum(
            subset["truth"] == subset["prediction"]
        ),
        "total": len(subset),
        "recall": np.mean(
            subset["truth"] == subset["prediction"]
        )
    })

class_results_df = pd.DataFrame(class_rows)

# =========================================================
# ABSTAIN METRICS
# =========================================================

abstain_pred = cases_df["prediction"] == "ABSTAIN"
abstain_true = cases_df["truth"] == "ABSTAIN"

abstain_precision = (
    np.sum(abstain_pred & abstain_true)
    / np.sum(abstain_pred)
)

abstain_recall = (
    np.sum(abstain_pred & abstain_true)
    / np.sum(abstain_true)
)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 80)
print("PER-SEED PERFORMANCE")
print("=" * 80)

print(
    seed_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 80)
print("AGGREGATE PERFORMANCE")
print("=" * 80)

print(f"Overall accuracy = {accuracy:.4f}")
print(f"False revision rate = {false_revision:.4f}")
print(f"Missed revision rate = {missed_revision:.4f}")
print(f"False recalibration rate = {false_recalibration:.4f}")
print(f"Unnecessary change rate = {unnecessary_change:.4f}")

print("\n" + "=" * 80)
print("AGGREGATE CLASS RECALL")
print("=" * 80)

print(
    class_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(
    pd.crosstab(
        cases_df["truth"],
        cases_df["prediction"]
    )
)

print("\n" + "=" * 80)
print("SAFETY CHECKS")
print("=" * 80)

false_revision_count = np.sum(
    (cases_df["prediction"] == "REVISE") &
    (cases_df["truth"] != "REVISE")
)

missed_revision_count = np.sum(
    (cases_df["truth"] == "REVISE") &
    (cases_df["prediction"] != "REVISE")
)

quality_gate_violations = np.sum(
    (cases_df["quality"] < QUALITY_THRESHOLD) &
    (cases_df["prediction"] != "ABSTAIN")
)

print("False structural revisions =", false_revision_count)
print("Missed structural revisions =", missed_revision_count)
print("Quality-gate violations =", quality_gate_violations)

print("\nABSTAIN precision =", f"{abstain_precision:.4f}")
print("ABSTAIN recall =", f"{abstain_recall:.4f}")

# =========================================================
# SEED VARIABILITY
# =========================================================

print("\n" + "=" * 80)
print("SEED-TO-SEED VARIABILITY")
print("=" * 80)

print(
    "Mean seed accuracy =",
    f"{seed_results_df['accuracy'].mean():.4f}"
)

print(
    "Std seed accuracy =",
    f"{seed_results_df['accuracy'].std(ddof=1):.4f}"
)

print(
    "Minimum seed accuracy =",
    f"{seed_results_df['accuracy'].min():.4f}"
)

print(
    "Maximum seed accuracy =",
    f"{seed_results_df['accuracy'].max():.4f}"
)

# =========================================================
# BENCHMARK SANITY CHECK
# =========================================================

print("\n" + "=" * 80)
print("BENCHMARK SANITY CHECK")
print("=" * 80)

rev = cases_df[cases_df["truth"] == "REVISE"]
abs_cases = cases_df[cases_df["truth"] == "ABSTAIN"]

print(
    "REVISE structural_gain range:",
    f"{rev['structural_gain'].min():.4f}",
    "to",
    f"{rev['structural_gain'].max():.4f}"
)

print(
    "REVISE quality range:",
    f"{rev['quality'].min():.4f}",
    "to",
    f"{rev['quality'].max():.4f}"
)

print(
    "ABSTAIN structural_gain range:",
    f"{abs_cases['structural_gain'].min():.4f}",
    "to",
    f"{abs_cases['structural_gain'].max():.4f}"
)

print(
    "ABSTAIN quality range:",
    f"{abs_cases['quality'].min():.4f}",
    "to",
    f"{abs_cases['quality'].max():.4f}"
)

print(
    "\nREVISE cases above structural threshold:",
    np.sum(
        rev["structural_gain"] > STRUCTURAL_THRESHOLD
    ),
    "/",
    len(rev)
)

print(
    "ABSTAIN cases below quality threshold:",
    np.sum(
        abs_cases["quality"] < QUALITY_THRESHOLD
    ),
    "/",
    len(abs_cases)
)

# =========================================================
# SAVE
# =========================================================

cases_df.to_csv(
    "step93_repaired_multiseed_cases.csv",
    index=False
)

seed_results_df.to_csv(
    "step93_repaired_multiseed_seed_results.csv",
    index=False
)

class_results_df.to_csv(
    "step93_repaired_multiseed_class_results.csv",
    index=False
)

print("\n" + "=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

print(
    "Step 93 repairs the Step 91 benchmark construction by explicitly "
    "assigning structural evidence to REVISE and ABSTAIN cases."
)

print(
    "The ETMR thresholds remain completely frozen."
)

print(
    "This is an independent multi-seed controlled benchmark and "
    "should be interpreted as a robustness test, not real-world validation."
)

print("\nSaved:")
print("step93_repaired_multiseed_cases.csv")
print("step93_repaired_multiseed_seed_results.csv")
print("step93_repaired_multiseed_class_results.csv")

print("\n" + "=" * 80)
print("STEP 93 COMPLETE")
print("=" * 80)
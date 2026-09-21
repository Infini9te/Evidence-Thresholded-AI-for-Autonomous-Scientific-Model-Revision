import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 95: 800-CASE BASELINE COMPARISON")
print("=" * 80)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("step93_repaired_multiseed_cases.csv")

ALPHA_LOWER = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

print("\nTotal cases =", len(df))

# =========================================================
# FROZEN ETMR
# =========================================================

def etmr(alpha, structural_gain, quality):

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    if structural_gain > STRUCTURAL_THRESHOLD:
        return "REVISE"

    if alpha < ALPHA_LOWER:
        return "RECALIBRATE"

    return "KEEP"


df["ETMR_prediction"] = df.apply(
    lambda r: etmr(
        r["alpha"],
        r["structural_gain"],
        r["quality"]
    ),
    axis=1
)

# =========================================================
# STRONGEST SIMPLE BASELINE
# STRUCTURAL + QUALITY GATE
# =========================================================

def baseline(structural_gain, quality):

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    if structural_gain > STRUCTURAL_THRESHOLD:
        return "REVISE"

    return "KEEP"


df["Baseline_prediction"] = df.apply(
    lambda r: baseline(
        r["structural_gain"],
        r["quality"]
    ),
    axis=1
)

# =========================================================
# METRICS
# =========================================================

def metrics(prediction):

    correct = (
        df["truth"] == df[prediction]
    )

    false_revision = (
        (df[prediction] == "REVISE") &
        (df["truth"] != "REVISE")
    )

    missed_revision = (
        (df["truth"] == "REVISE") &
        (df[prediction] != "REVISE")
    )

    false_recalibration = (
        (df[prediction] == "RECALIBRATE") &
        (df["truth"] != "RECALIBRATE")
    )

    unnecessary_change = (
        (df["truth"] == "KEEP") &
        (df[prediction] != "KEEP")
    )

    abstain_pred = (
        df[prediction] == "ABSTAIN"
    )

    abstain_true = (
        df["truth"] == "ABSTAIN"
    )

    abstain_precision = (
        np.sum(abstain_pred & abstain_true)
        / np.sum(abstain_pred)
        if np.sum(abstain_pred) > 0
        else np.nan
    )

    abstain_recall = (
        np.sum(abstain_pred & abstain_true)
        / np.sum(abstain_true)
    )

    return {
        "accuracy": correct.mean(),
        "false_revision_rate": false_revision.mean(),
        "missed_revision_rate": missed_revision.mean(),
        "false_recalibration_rate": false_recalibration.mean(),
        "unnecessary_change_rate": unnecessary_change.mean(),
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall
    }


etmr_metrics = metrics("ETMR_prediction")
baseline_metrics = metrics("Baseline_prediction")

# =========================================================
# PRINT MAIN COMPARISON
# =========================================================

print("\n" + "=" * 80)
print("OVERALL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame(
    {
        "ETMR": etmr_metrics,
        "Structural+Quality": baseline_metrics
    }
).T

print(
    comparison.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

# =========================================================
# CLASS RECALL
# =========================================================

print("\n" + "=" * 80)
print("CLASS RECALL")
print("=" * 80)

rows = []

for cls in [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]:

    true_class = df["truth"] == cls

    etmr_recall = np.mean(
        df.loc[true_class, "ETMR_prediction"] == cls
    )

    baseline_recall = np.mean(
        df.loc[true_class, "Baseline_prediction"] == cls
    )

    rows.append({
        "class": cls,
        "ETMR_recall": etmr_recall,
        "Baseline_recall": baseline_recall
    })

class_df = pd.DataFrame(rows)

print(
    class_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# =========================================================
# CONFUSION MATRICES
# =========================================================

print("\n" + "=" * 80)
print("ETMR CONFUSION MATRIX")
print("=" * 80)

print(
    pd.crosstab(
        df["truth"],
        df["ETMR_prediction"]
    )
)

print("\n" + "=" * 80)
print("BASELINE CONFUSION MATRIX")
print("=" * 80)

print(
    pd.crosstab(
        df["truth"],
        df["Baseline_prediction"]
    )
)

# =========================================================
# PAIRED ACCURACY COMPARISON
# =========================================================

etmr_correct = (
    df["truth"] == df["ETMR_prediction"]
)

baseline_correct = (
    df["truth"] == df["Baseline_prediction"]
)

etmr_wins = np.sum(
    etmr_correct & ~baseline_correct
)

baseline_wins = np.sum(
    baseline_correct & ~etmr_correct
)

ties = np.sum(
    etmr_correct & baseline_correct
)

print("\n" + "=" * 80)
print("PAIRED OUTCOME COMPARISON")
print("=" * 80)

print("ETMR wins =", etmr_wins)
print("Baseline wins =", baseline_wins)
print("Ties =", ties)

observed_advantage = (
    etmr_correct.mean()
    - baseline_correct.mean()
)

print(
    "Observed accuracy advantage =",
    f"{observed_advantage:.4f}"
)

# =========================================================
# BOOTSTRAP ADVANTAGE
# =========================================================

rng = np.random.default_rng(20260950)

advantages = (
    etmr_correct.astype(int)
    - baseline_correct.astype(int)
).to_numpy()

boot = []

for _ in range(10000):

    sample = rng.choice(
        advantages,
        size=len(advantages),
        replace=True
    )

    boot.append(
        np.mean(sample)
    )

boot = np.array(boot)

ci_low = np.percentile(boot, 2.5)
ci_high = np.percentile(boot, 97.5)

p_etmr_better = np.mean(
    boot > 0
)

print("\n" + "=" * 80)
print("BOOTSTRAP ACCURACY ADVANTAGE")
print("=" * 80)

print(
    "Mean bootstrap advantage =",
    f"{boot.mean():.4f}"
)

print(
    "95% CI =",
    f"[{ci_low:.4f}, {ci_high:.4f}]"
)

print(
    "P(ETMR > baseline) =",
    f"{p_etmr_better:.4f}"
)

# =========================================================
# ERROR REDUCTION
# =========================================================

etmr_errors = np.sum(~etmr_correct)
baseline_errors = np.sum(~baseline_correct)

if baseline_errors > 0:

    error_reduction = (
        baseline_errors - etmr_errors
    ) / baseline_errors

else:

    error_reduction = np.nan

print("\n" + "=" * 80)
print("ERROR REDUCTION")
print("=" * 80)

print("ETMR errors =", etmr_errors)
print("Baseline errors =", baseline_errors)

print(
    "Relative error reduction =",
    f"{error_reduction:.4f}"
)

# =========================================================
# STRUCTURAL SAFETY
# =========================================================

print("\n" + "=" * 80)
print("STRUCTURAL SAFETY")
print("=" * 80)

for name, col in [
    ("ETMR", "ETMR_prediction"),
    ("Baseline", "Baseline_prediction")
]:

    false_rev = np.sum(
        (df[col] == "REVISE") &
        (df["truth"] != "REVISE")
    )

    missed_rev = np.sum(
        (df["truth"] == "REVISE") &
        (df[col] != "REVISE")
    )

    print(
        f"{name}: false revisions = {false_rev}, "
        f"missed revisions = {missed_rev}"
    )

# =========================================================
# SAVE
# =========================================================

comparison.to_csv(
    "step95_overall_comparison.csv"
)

class_df.to_csv(
    "step95_class_recall_comparison.csv",
    index=False
)

summary = pd.DataFrame([{
    "n": len(df),
    "ETMR_accuracy": etmr_metrics["accuracy"],
    "Baseline_accuracy": baseline_metrics["accuracy"],
    "accuracy_advantage": observed_advantage,
    "bootstrap_ci_lower": ci_low,
    "bootstrap_ci_upper": ci_high,
    "p_etmr_better": p_etmr_better,
    "ETMR_errors": etmr_errors,
    "Baseline_errors": baseline_errors,
    "relative_error_reduction": error_reduction
}])

summary.to_csv(
    "step95_statistical_baseline_comparison.csv",
    index=False
)

# =========================================================
# INTERPRETATION
# =========================================================

print("\n" + "=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

print(
    "Step 95 compares frozen ETMR against the strongest "
    "simple structural-plus-quality baseline on the same "
    "800 independent controlled cases."
)

print(
    "The baseline does not contain parameter evidence and "
    "therefore cannot explicitly distinguish KEEP from "
    "RECALIBRATE."
)

print(
    "No ETMR threshold was fitted or modified for this comparison."
)

print(
    "Results remain controlled synthetic evidence and do not "
    "establish universal real-world performance."
)

print("\nSaved:")
print("step95_overall_comparison.csv")
print("step95_class_recall_comparison.csv")
print("step95_statistical_baseline_comparison.csv")

print("\n" + "=" * 80)
print("STEP 95 COMPLETE")
print("=" * 80)
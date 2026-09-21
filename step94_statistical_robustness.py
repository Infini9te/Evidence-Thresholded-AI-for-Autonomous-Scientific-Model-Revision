import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 94: STATISTICAL ROBUSTNESS OF MULTI-SEED GENERALIZATION")
print("=" * 80)

# =========================================================
# LOAD STEP 93 RESULTS
# =========================================================

df = pd.read_csv("step93_repaired_multiseed_cases.csv")

print("\nTotal cases =", len(df))

# =========================================================
# FROZEN POLICY — FOR REFERENCE ONLY
# =========================================================

print("\n" + "=" * 80)
print("FROZEN POLICY")
print("=" * 80)

print("Alpha lower boundary = 1.039534")
print("Structural threshold = 0.388612")
print("Quality threshold = 0.80")

# =========================================================
# BASIC METRICS
# =========================================================

correct = (
    df["truth"] == df["prediction"]
).astype(int)

accuracy = correct.mean()

false_revision = (
    (df["prediction"] == "REVISE") &
    (df["truth"] != "REVISE")
).astype(int)

missed_revision = (
    (df["truth"] == "REVISE") &
    (df["prediction"] != "REVISE")
).astype(int)

unnecessary_change = (
    (df["truth"] == "KEEP") &
    (df["prediction"] != "KEEP")
).astype(int)

# =========================================================
# BOOTSTRAP FUNCTION
# =========================================================

rng = np.random.default_rng(20260940)

def bootstrap_ci(values, n_boot=10000, alpha=0.05):

    values = np.asarray(values)
    n = len(values)

    boot_means = np.empty(n_boot)

    for i in range(n_boot):

        sample = rng.choice(
            values,
            size=n,
            replace=True
        )

        boot_means[i] = np.mean(sample)

    lower = np.percentile(
        boot_means,
        100 * alpha / 2
    )

    upper = np.percentile(
        boot_means,
        100 * (1 - alpha / 2)
    )

    return (
        np.mean(values),
        lower,
        upper
    )

# =========================================================
# OVERALL ACCURACY
# =========================================================

acc_mean, acc_low, acc_high = bootstrap_ci(correct)

print("\n" + "=" * 80)
print("OVERALL ACCURACY")
print("=" * 80)

print(f"Observed accuracy = {accuracy:.4f}")
print(
    f"Bootstrap 95% CI = [{acc_low:.4f}, {acc_high:.4f}]"
)

# =========================================================
# CLASS RECALL
# =========================================================

print("\n" + "=" * 80)
print("CLASS RECALL WITH 95% BOOTSTRAP CI")
print("=" * 80)

classes = [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]

class_results = []

for cls in classes:

    subset = df[df["truth"] == cls]

    values = (
        subset["prediction"] == cls
    ).astype(int)

    observed = values.mean()

    mean, low, high = bootstrap_ci(values)

    class_results.append({
        "class": cls,
        "n": len(values),
        "recall": observed,
        "ci_lower": low,
        "ci_upper": high
    })

    print(
        f"{cls:15s} "
        f"n={len(values):3d} "
        f"recall={observed:.4f} "
        f"95% CI=[{low:.4f}, {high:.4f}]"
    )

class_results_df = pd.DataFrame(class_results)

# =========================================================
# SAFETY METRICS
# =========================================================

print("\n" + "=" * 80)
print("SAFETY METRICS WITH 95% BOOTSTRAP CI")
print("=" * 80)

metrics = {
    "False revision rate": false_revision,
    "Missed revision rate": missed_revision,
    "Unnecessary change rate": unnecessary_change
}

safety_rows = []

for name, values in metrics.items():

    observed = values.mean()

    mean, low, high = bootstrap_ci(values)

    safety_rows.append({
        "metric": name,
        "observed": observed,
        "ci_lower": low,
        "ci_upper": high
    })

    print(
        f"{name:28s} "
        f"observed={observed:.4f} "
        f"95% CI=[{low:.4f}, {high:.4f}]"
    )

safety_df = pd.DataFrame(safety_rows)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(
    pd.crosstab(
        df["truth"],
        df["prediction"]
    )
)

# =========================================================
# SEED-LEVEL RESULTS
# =========================================================

seed_rows = []

for seed, g in df.groupby("seed"):

    seed_accuracy = np.mean(
        g["truth"] == g["prediction"]
    )

    seed_rows.append({
        "seed": seed,
        "accuracy": seed_accuracy
    })

seed_df = pd.DataFrame(seed_rows)

print("\n" + "=" * 80)
print("SEED-LEVEL ACCURACY")
print("=" * 80)

print(
    seed_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nMean seed accuracy =",
      f"{seed_df['accuracy'].mean():.4f}")

print("Std seed accuracy =",
      f"{seed_df['accuracy'].std(ddof=1):.4f}")

print("Minimum seed accuracy =",
      f"{seed_df['accuracy'].min():.4f}")

print("Maximum seed accuracy =",
      f"{seed_df['accuracy'].max():.4f}")

# =========================================================
# ERROR ANALYSIS
# =========================================================

errors = df[
    df["truth"] != df["prediction"]
].copy()

print("\n" + "=" * 80)
print("ERROR ANALYSIS")
print("=" * 80)

print("Total errors =", len(errors))

if len(errors) > 0:

    print("\nError confusion:")
    print(
        pd.crosstab(
            errors["truth"],
            errors["prediction"]
        )
    )

    print("\nError truth classes:")
    print(
        errors["truth"]
        .value_counts()
        .sort_index()
    )

# =========================================================
# STRUCTURAL SAFETY
# =========================================================

false_revision_count = np.sum(
    (df["prediction"] == "REVISE") &
    (df["truth"] != "REVISE")
)

missed_revision_count = np.sum(
    (df["truth"] == "REVISE") &
    (df["prediction"] != "REVISE")
)

print("\n" + "=" * 80)
print("STRUCTURAL SAFETY")
print("=" * 80)

print(
    "False structural revisions =",
    false_revision_count
)

print(
    "Missed structural revisions =",
    missed_revision_count
)

if false_revision_count == 0:
    print("PASS: zero false structural revisions.")

if missed_revision_count == 0:
    print("PASS: zero missed structural revisions.")

# =========================================================
# SAVE RESULTS
# =========================================================

class_results_df.to_csv(
    "step94_class_bootstrap_results.csv",
    index=False
)

safety_df.to_csv(
    "step94_safety_bootstrap_results.csv",
    index=False
)

seed_df.to_csv(
    "step94_seed_accuracy_results.csv",
    index=False
)

summary = pd.DataFrame([{
    "n": len(df),
    "accuracy": accuracy,
    "accuracy_ci_lower": acc_low,
    "accuracy_ci_upper": acc_high,
    "false_revision_rate": false_revision.mean(),
    "missed_revision_rate": missed_revision.mean(),
    "unnecessary_change_rate": unnecessary_change.mean(),
    "mean_seed_accuracy": seed_df["accuracy"].mean(),
    "std_seed_accuracy": seed_df["accuracy"].std(ddof=1)
}])

summary.to_csv(
    "step94_statistical_robustness_summary.csv",
    index=False
)

# =========================================================
# INTERPRETATION
# =========================================================

print("\n" + "=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

print(
    "Step 94 quantifies statistical uncertainty around the "
    "Step 93 independent multi-seed evaluation."
)

print(
    "The ETMR thresholds remain frozen and are not optimized "
    "using the Step 93 or Step 94 cases."
)

print(
    "The bootstrap intervals quantify sampling uncertainty; "
    "they do not establish real-world validity."
)

print("\nSaved:")
print("step94_class_bootstrap_results.csv")
print("step94_safety_bootstrap_results.csv")
print("step94_seed_accuracy_results.csv")
print("step94_statistical_robustness_summary.csv")

print("\n" + "=" * 80)
print("STEP 94 COMPLETE")
print("=" * 80)
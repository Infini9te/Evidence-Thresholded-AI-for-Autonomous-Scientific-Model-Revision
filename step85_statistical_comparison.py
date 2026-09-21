import pandas as pd
import numpy as np
from scipy.stats import binomtest

# ============================================================
# ETMR — STEP 85: STATISTICAL BASELINE COMPARISON
# ============================================================

FILE = "step82_final_frozen_evaluation_results.csv"

STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

N_BOOTSTRAP = 10000
SEED = 20260914

rng = np.random.default_rng(SEED)

df = pd.read_csv(FILE)

truth = df["truth"].to_numpy()

# ------------------------------------------------------------
# Frozen ETMR prediction
# ------------------------------------------------------------

etmr = df["predicted"].to_numpy()

# ------------------------------------------------------------
# Strongest simple baseline from Step 84:
#
# Structural evidence + quality gate
#
# Low quality -> ABSTAIN
# High quality + strong structural improvement -> REVISE
# Otherwise -> KEEP
# ------------------------------------------------------------

baseline = np.where(
    df["quality"].to_numpy() < QUALITY_THRESHOLD,
    "ABSTAIN",
    np.where(
        df["structural_relative_gain"].to_numpy()
        > STRUCTURAL_THRESHOLD,
        "REVISE",
        "KEEP"
    )
)

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

etmr_correct = (etmr == truth)
baseline_correct = (baseline == truth)

etmr_accuracy = etmr_correct.mean()
baseline_accuracy = baseline_correct.mean()

observed_difference = etmr_accuracy - baseline_accuracy

print("=" * 80)
print("ETMR — STEP 85: STATISTICAL BASELINE COMPARISON")
print("=" * 80)

print(f"\nLoaded: {FILE}")
print(f"Number of cases = {len(df)}")

print("\n" + "=" * 80)
print("OBSERVED PERFORMANCE")
print("=" * 80)

print(f"\nFrozen ETMR accuracy = {etmr_accuracy:.4f}")
print(f"Best baseline accuracy = {baseline_accuracy:.4f}")
print(f"Observed accuracy advantage = {observed_difference:.4f}")

# ------------------------------------------------------------
# Paired win / tie / loss
# ------------------------------------------------------------

etmr_wins = np.sum(etmr_correct & ~baseline_correct)
baseline_wins = np.sum(~etmr_correct & baseline_correct)
ties = np.sum(etmr_correct == baseline_correct)

print("\n" + "=" * 80)
print("PAIRED CASE COMPARISON")
print("=" * 80)

print(f"ETMR wins    = {etmr_wins}")
print(f"Baseline wins = {baseline_wins}")
print(f"Ties         = {ties}")

# ------------------------------------------------------------
# Bootstrap paired accuracy difference
# ------------------------------------------------------------

differences = []

n = len(df)

for _ in range(N_BOOTSTRAP):

    idx = rng.integers(0, n, size=n)

    etmr_acc = etmr_correct[idx].mean()
    base_acc = baseline_correct[idx].mean()

    differences.append(etmr_acc - base_acc)

differences = np.asarray(differences)

ci_low = np.percentile(differences, 2.5)
ci_high = np.percentile(differences, 97.5)

prob_etmr_better = np.mean(differences > 0)

print("\n" + "=" * 80)
print("BOOTSTRAP ACCURACY ADVANTAGE")
print("=" * 80)

print(f"Bootstrap repetitions = {N_BOOTSTRAP}")
print(f"Mean advantage = {differences.mean():.4f}")
print(f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
print(f"P(ETMR > baseline) = {prob_etmr_better:.4f}")

# ------------------------------------------------------------
# McNemar-style paired comparison
#
# b = baseline correct, ETMR wrong
# c = ETMR correct, baseline wrong
# ------------------------------------------------------------

b = np.sum(~etmr_correct & baseline_correct)
c = np.sum(etmr_correct & ~baseline_correct)

print("\n" + "=" * 80)
print("PAIRED ERROR TEST")
print("=" * 80)

print(f"Baseline correct / ETMR wrong = {b}")
print(f"ETMR correct / baseline wrong = {c}")

if (b + c) > 0:

    # Exact two-sided McNemar test using binomial test
    smaller = min(b, c)

    mcnemar_result = binomtest(
        smaller,
        n=b + c,
        p=0.5,
        alternative="two-sided"
    )

    p_value = mcnemar_result.pvalue

else:

    p_value = 1.0

print(f"Exact paired-test p-value = {p_value:.6f}")

# ------------------------------------------------------------
# Class-level comparison
# ------------------------------------------------------------

classes = ["KEEP", "RECALIBRATE", "REVISE", "ABSTAIN"]

print("\n" + "=" * 80)
print("CLASS-LEVEL PERFORMANCE")
print("=" * 80)

rows = []

for cls in classes:

    mask = truth == cls

    etmr_recall = np.mean(etmr[mask] == cls)
    baseline_recall = np.mean(baseline[mask] == cls)

    rows.append({
        "class": cls,
        "ETMR_recall": etmr_recall,
        "baseline_recall": baseline_recall,
        "difference": etmr_recall - baseline_recall
    })

class_df = pd.DataFrame(rows)

print(
    class_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# Error-cost comparison
# ------------------------------------------------------------

def error_cost(truth, pred):

    cost = 0

    for t, p in zip(truth, pred):

        if p == "REVISE" and t != "REVISE":
            cost += 5

        elif t == "REVISE" and p != "REVISE":
            cost += 3

        elif p == "RECALIBRATE" and t != "RECALIBRATE":
            cost += 2

        elif t == "KEEP" and p in ["RECALIBRATE", "REVISE"]:
            cost += 2

        elif p == "ABSTAIN" and t != "ABSTAIN":
            cost += 1

    return cost


etmr_cost = error_cost(truth, etmr)
baseline_cost = error_cost(truth, baseline)

print("\n" + "=" * 80)
print("ILLUSTRATIVE ERROR-COST")
print("=" * 80)

print(f"Frozen ETMR cost = {etmr_cost}")
print(f"Baseline cost    = {baseline_cost}")
print(f"Cost reduction   = {baseline_cost - etmr_cost}")

# ------------------------------------------------------------
# Safety comparison
# ------------------------------------------------------------

etmr_false_revision = np.mean(
    (truth != "REVISE") & (etmr == "REVISE")
)

baseline_false_revision = np.mean(
    (truth != "REVISE") & (baseline == "REVISE")
)

etmr_missed_revision = np.mean(
    (truth == "REVISE") & (etmr != "REVISE")
)

baseline_missed_revision = np.mean(
    (truth == "REVISE") & (baseline != "REVISE")
)

print("\n" + "=" * 80)
print("SCIENTIFIC SAFETY COMPARISON")
print("=" * 80)

print(
    f"\nETMR false revision rate = "
    f"{etmr_false_revision:.4f}"
)

print(
    f"Baseline false revision rate = "
    f"{baseline_false_revision:.4f}"
)

print(
    f"\nETMR missed revision rate = "
    f"{etmr_missed_revision:.4f}"
)

print(
    f"Baseline missed revision rate = "
    f"{baseline_missed_revision:.4f}"
)

# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

if observed_difference > 0:
    print(
        f"\nETMR improves accuracy by "
        f"{observed_difference * 100:.1f} percentage points "
        f"over the strongest simple baseline."
    )

if ci_low > 0:
    print(
        "PASS: The bootstrap 95% confidence interval "
        "for the accuracy advantage excludes zero."
    )
else:
    print(
        "NOTE: The bootstrap 95% confidence interval "
        "includes zero; superiority is not statistically "
        "established despite the observed advantage."
    )

if p_value < 0.05:
    print(
        "PASS: The exact paired test indicates a "
        "statistically significant difference."
    )
else:
    print(
        "NOTE: The exact paired test does not reach "
        "the conventional 0.05 significance level."
    )

if baseline_false_revision > etmr_false_revision:
    print(
        "\nETMR has a lower false-revision rate "
        "than the baseline."
    )

if baseline_missed_revision > etmr_missed_revision:
    print(
        "ETMR has a lower missed-revision rate "
        "than the baseline."
    )

print(
    "\nIMPORTANT:"
    "\nThis benchmark is controlled and synthetic."
    "\nThe statistical test quantifies robustness on this "
    "benchmark; it does not establish real-world accuracy."
)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

summary = pd.DataFrame([{
    "ETMR_accuracy": etmr_accuracy,
    "baseline_accuracy": baseline_accuracy,
    "observed_accuracy_advantage": observed_difference,
    "bootstrap_mean_advantage": differences.mean(),
    "bootstrap_CI_low": ci_low,
    "bootstrap_CI_high": ci_high,
    "P_ETMR_better": prob_etmr_better,
    "ETMR_wins": etmr_wins,
    "baseline_wins": baseline_wins,
    "ties": ties,
    "paired_test_p_value": p_value,
    "ETMR_error_cost": etmr_cost,
    "baseline_error_cost": baseline_cost,
    "ETMR_false_revision_rate": etmr_false_revision,
    "baseline_false_revision_rate": baseline_false_revision,
    "ETMR_missed_revision_rate": etmr_missed_revision,
    "baseline_missed_revision_rate": baseline_missed_revision
}])

summary.to_csv(
    "step85_statistical_comparison_results.csv",
    index=False
)

class_df.to_csv(
    "step85_class_level_comparison.csv",
    index=False
)

print("\nSaved:")
print("step85_statistical_comparison_results.csv")
print("step85_class_level_comparison.csv")

print("\n" + "=" * 80)
print("STEP 85 COMPLETE")
print("=" * 80)
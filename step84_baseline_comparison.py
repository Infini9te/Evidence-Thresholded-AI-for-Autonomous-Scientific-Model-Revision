import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================================================
# ETMR — STEP 84: BASELINE COMPARISON
# ============================================================

FILE = "step82_final_frozen_evaluation_results.csv"

# Frozen ETMR thresholds from previous steps
ALPHA_LOWER = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

df = pd.read_csv(FILE)

print("=" * 80)
print("ETMR — STEP 84: BASELINE COMPARISON")
print("=" * 80)

print(f"\nLoaded: {FILE}")
print(f"Number of cases = {len(df)}")

# ------------------------------------------------------------
# Helper metrics
# ------------------------------------------------------------

def metrics(name, truth, pred):
    truth = np.asarray(truth)
    pred = np.asarray(pred)

    acc = accuracy_score(truth, pred)

    false_revision = np.mean(
        (truth != "REVISE") & (pred == "REVISE")
    )

    missed_revision = np.mean(
        (truth == "REVISE") & (pred != "REVISE")
    )

    false_recalibration = np.mean(
        (truth != "RECALIBRATE") & (pred == "RECALIBRATE")
    )

    unnecessary_change = np.mean(
        (truth == "KEEP") &
        np.isin(pred, ["RECALIBRATE", "REVISE"])
    )

    abstain_true = truth == "ABSTAIN"
    abstain_pred = pred == "ABSTAIN"

    if abstain_pred.sum() > 0:
        abstain_precision = np.mean(
            truth[abstain_pred] == "ABSTAIN"
        )
    else:
        abstain_precision = np.nan

    if abstain_true.sum() > 0:
        abstain_recall = np.mean(
            pred[abstain_true] == "ABSTAIN"
        )
    else:
        abstain_recall = np.nan

    return {
        "method": name,
        "accuracy": acc,
        "false_revision_rate": false_revision,
        "missed_revision_rate": missed_revision,
        "false_recalibration_rate": false_recalibration,
        "unnecessary_change_rate": unnecessary_change,
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall
    }

# ------------------------------------------------------------
# Baseline 1
# Residual / structural-error threshold
#
# If structural gain is large -> REVISE
# Otherwise -> KEEP
#
# This represents a simple "large discrepancy = revise"
# strategy and ignores parameter-vs-structural competition,
# provenance, and abstention.
# ------------------------------------------------------------

pred_residual = np.where(
    df["structural_relative_gain"] > STRUCTURAL_THRESHOLD,
    "REVISE",
    "KEEP"
)

# ------------------------------------------------------------
# Baseline 2
# Structural-only rule with quality gate
#
# High discrepancy + high quality -> REVISE
# Low quality -> ABSTAIN
# Otherwise -> KEEP
#
# This tests whether the quality gate alone explains ETMR's
# performance.
# ------------------------------------------------------------

pred_structural_quality = np.where(
    df["quality"] < QUALITY_THRESHOLD,
    "ABSTAIN",
    np.where(
        df["structural_relative_gain"] > STRUCTURAL_THRESHOLD,
        "REVISE",
        "KEEP"
    )
)

# ------------------------------------------------------------
# Baseline 3
# Parameter-only rule
#
# alpha below frozen KEEP interval -> RECALIBRATE
# otherwise -> KEEP
#
# Ignores structural evidence and abstention.
# ------------------------------------------------------------

pred_parameter = np.where(
    df["alpha_case"] < ALPHA_LOWER,
    "RECALIBRATE",
    "KEEP"
)

# ------------------------------------------------------------
# Baseline 4
# Parameter + quality
#
# Low quality -> ABSTAIN
# Otherwise parameter deviation -> RECALIBRATE
# Otherwise KEEP
# ------------------------------------------------------------

pred_parameter_quality = np.where(
    df["quality"] < QUALITY_THRESHOLD,
    "ABSTAIN",
    np.where(
        df["alpha_case"] < ALPHA_LOWER,
        "RECALIBRATE",
        "KEEP"
    )
)

# ------------------------------------------------------------
# Baseline 5
# Improvement-only rule
#
# Large structural counterfactual improvement -> REVISE
# Otherwise KEEP.
#
# No distinction between parameter and structural change.
# ------------------------------------------------------------

pred_improvement = np.where(
    df["structural_relative_gain"] > STRUCTURAL_THRESHOLD,
    "REVISE",
    "KEEP"
)

# ------------------------------------------------------------
# ETMR frozen prediction
# ------------------------------------------------------------

pred_etmr = df["predicted"].values


# ------------------------------------------------------------
# Collect results
# ------------------------------------------------------------

results = []

results.append(
    metrics(
        "Residual/Structural Threshold",
        df["truth"].values,
        pd.Series(pred_residual)
    )
)

results.append(
    metrics(
        "Structural + Quality Gate",
        df["truth"].values,
        pd.Series(pred_structural_quality)
    )
)

results.append(
    metrics(
        "Parameter-Only",
        df["truth"].values,
        pd.Series(pred_parameter)
    )
)

results.append(
    metrics(
        "Parameter + Quality",
        df["truth"].values,
        pd.Series(pred_parameter_quality)
    )
)

results.append(
    metrics(
        "Improvement-Only",
        df["truth"].values,
        pd.Series(pred_improvement)
    )
)

results.append(
    metrics(
        "Frozen ETMR",
        df["truth"].values,
        pd.Series(pred_etmr)
    )
)

results_df = pd.DataFrame(results)


# ------------------------------------------------------------
# Print comparison
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("BASELINE PERFORMANCE")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# Confusion matrices
# ------------------------------------------------------------

labels = ["KEEP", "RECALIBRATE", "REVISE", "ABSTAIN"]

for name, pred in [
    ("Residual/Structural Threshold", pred_residual),
    ("Structural + Quality Gate", pred_structural_quality),
    ("Parameter-Only", pred_parameter),
    ("Parameter + Quality", pred_parameter_quality),
    ("Improvement-Only", pred_improvement),
    ("Frozen ETMR", pred_etmr),
]:

    cm = confusion_matrix(
        df["truth"],
        pred,
        labels=labels
    )

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print(cm_df)


# ------------------------------------------------------------
# Error-cost comparison
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ILLUSTRATIVE SCIENTIFIC ERROR COST")
print("=" * 80)

print("""
Weights:
False REVISE       = 5
Missed REVISE      = 3
False RECALIBRATE  = 2
Unnecessary change = 2
Incorrect ABSTAIN  = 1
""")

def error_cost(truth, pred):

    truth = np.asarray(truth)
    pred = np.asarray(pred)

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


cost_rows = []

for name, pred in [
    ("Residual/Structural Threshold", pred_residual),
    ("Structural + Quality Gate", pred_structural_quality),
    ("Parameter-Only", pred_parameter),
    ("Parameter + Quality", pred_parameter_quality),
    ("Improvement-Only", pred_improvement),
    ("Frozen ETMR", pred_etmr),
]:

    cost_rows.append({
        "method": name,
        "total_error_cost": error_cost(
            df["truth"].values,
            pred
        )
    })

cost_df = pd.DataFrame(cost_rows)

print(cost_df.to_string(index=False))


# ------------------------------------------------------------
# Key scientific comparison
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

etmr = results_df[
    results_df["method"] == "Frozen ETMR"
].iloc[0]

print(
    f"\nFrozen ETMR accuracy = "
    f"{etmr['accuracy']:.4f}"
)

print(
    f"Frozen ETMR false revision rate = "
    f"{etmr['false_revision_rate']:.4f}"
)

print(
    f"Frozen ETMR missed revision rate = "
    f"{etmr['missed_revision_rate']:.4f}"
)

print(
    f"Frozen ETMR unnecessary change rate = "
    f"{etmr['unnecessary_change_rate']:.4f}"
)

best_baseline = results_df[
    results_df["method"] != "Frozen ETMR"
].sort_values(
    "accuracy",
    ascending=False
).iloc[0]

print(
    f"\nBest baseline by accuracy = "
    f"{best_baseline['method']}"
)

print(
    f"Best baseline accuracy = "
    f"{best_baseline['accuracy']:.4f}"
)

print(
    f"\nETMR accuracy advantage over best baseline = "
    f"{etmr['accuracy'] - best_baseline['accuracy']:.4f}"
)


if etmr["false_revision_rate"] == 0:
    print(
        "\nPASS: ETMR produces zero false structural revisions."
    )

if etmr["missed_revision_rate"] == 0:
    print(
        "PASS: ETMR misses zero planted structural revisions."
    )

if etmr["unnecessary_change_rate"] == 0:
    print(
        "PASS: ETMR makes zero unnecessary changes to KEEP cases."
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results_df.to_csv(
    "step84_baseline_comparison_results.csv",
    index=False
)

cost_df.to_csv(
    "step84_baseline_error_cost_results.csv",
    index=False
)

print("\nSaved:")
print("step84_baseline_comparison_results.csv")
print("step84_baseline_error_cost_results.csv")

print("\n" + "=" * 80)
print("STEP 84 COMPLETE")
print("=" * 80)
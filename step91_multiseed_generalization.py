import numpy as np
import pandas as pd

# ============================================================
# ETMR — STEP 91
# MULTI-SEED INDEPENDENT GENERALIZATION
# ============================================================

SEEDS = [
    20260921,
    20260922,
    20260923,
    20260924,
    20260925,
    20260926,
    20260927,
    20260928,
    20260929,
    20260930
]

N_PER_CLASS = 20

# ============================================================
# FROZEN POLICY — NEVER CHANGE
# ============================================================

ALPHA_LOWER = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80


# ============================================================
# DECISION FUNCTION
# ============================================================

def etmr_decision(alpha, structural_gain, quality):

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    if structural_gain > STRUCTURAL_THRESHOLD:
        return "REVISE"

    if alpha < ALPHA_LOWER:
        return "RECALIBRATE"

    return "KEEP"


# ============================================================
# RUN ONE INDEPENDENT BENCHMARK
# ============================================================

def run_seed(seed):

    rng = np.random.default_rng(seed)

    rows = []

    # --------------------------------------------------------
    # KEEP
    # --------------------------------------------------------

    for i in range(N_PER_CLASS):

        alpha = rng.normal(
            loc=1.0405,
            scale=0.00065
        )

        rows.append({
            "case_id": f"S{seed}_KEEP_{i+1:02d}",
            "truth": "KEEP",
            "alpha": alpha,
            "structural_gain": 0.0,
            "quality": rng.uniform(0.82, 1.00)
        })


    # --------------------------------------------------------
    # RECALIBRATE
    # --------------------------------------------------------

    for i in range(N_PER_CLASS):

        alpha = rng.uniform(
            0.90,
            1.03
        )

        rows.append({
            "case_id": f"S{seed}_RECALIBRATE_{i+1:02d}",
            "truth": "RECALIBRATE",
            "alpha": alpha,
            "structural_gain": 0.0,
            "quality": rng.uniform(0.82, 1.00)
        })


    # --------------------------------------------------------
    # REVISE
    # --------------------------------------------------------

    for i in range(N_PER_CLASS):

        alpha = rng.uniform(
            1.70,
            2.60
        )

        rows.append({
            "case_id": f"S{seed}_REVISE_{i+1:02d}",
            "truth": "REVISE",
            "alpha": rng.uniform(0.45, 0.95),
            "quality": rng.uniform(0.82, 1.00)
        })


    # --------------------------------------------------------
    # ABSTAIN
    # --------------------------------------------------------

    for i in range(N_PER_CLASS):

        alpha = rng.uniform(
            1.70,
            2.60
        )

        rows.append({
            "case_id": f"S{seed}_ABSTAIN_{i+1:02d}",
            "truth": "ABSTAIN",
            "alpha": rng.uniform(0.45, 0.95),
            "quality": rng.uniform(0.20, 0.79)
        })


    df = pd.DataFrame(rows)

    df["prediction"] = df.apply(
        lambda r: etmr_decision(
            r["alpha"],
            r["structural_gain"],
            r["quality"]
        ),
        axis=1
    )

    df["correct"] = (
        df["truth"] == df["prediction"]
    )

    false_revision = (
        (df["prediction"] == "REVISE") &
        (df["truth"] != "REVISE")
    )

    missed_revision = (
        (df["truth"] == "REVISE") &
        (df["prediction"] != "REVISE")
    )

    false_recalibration = (
        (df["prediction"] == "RECALIBRATE") &
        (df["truth"] != "RECALIBRATE")
    )

    unnecessary_change = (
        (df["truth"] == "KEEP") &
        (df["prediction"] != "KEEP")
    )

    true_abstain = (
        df["truth"] == "ABSTAIN"
    )

    pred_abstain = (
        df["prediction"] == "ABSTAIN"
    )

    abstain_recall = (
        (true_abstain & pred_abstain).sum()
        / true_abstain.sum()
    )

    abstain_precision = (
        (true_abstain & pred_abstain).sum()
        / pred_abstain.sum()
        if pred_abstain.sum() > 0
        else np.nan
    )

    class_recall = {}

    for cls in [
        "KEEP",
        "RECALIBRATE",
        "REVISE",
        "ABSTAIN"
    ]:

        mask = (
            df["truth"] == cls
        )

        class_recall[cls] = (
            df.loc[mask, "prediction"] == cls
        ).mean()

    return df, {
        "seed": seed,
        "n": len(df),
        "accuracy": df["correct"].mean(),
        "false_revision_rate": false_revision.mean(),
        "missed_revision_rate": missed_revision.mean(),
        "false_recalibration_rate": false_recalibration.mean(),
        "unnecessary_change_rate": unnecessary_change.mean(),
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "KEEP_recall": class_recall["KEEP"],
        "RECALIBRATE_recall": class_recall["RECALIBRATE"],
        "REVISE_recall": class_recall["REVISE"],
        "ABSTAIN_recall": class_recall["ABSTAIN"]
    }


# ============================================================
# RUN ALL SEEDS
# ============================================================

all_cases = []
summaries = []

for seed in SEEDS:

    cases, summary = run_seed(seed)

    all_cases.append(cases)
    summaries.append(summary)


cases_all = pd.concat(
    all_cases,
    ignore_index=True
)

summary_df = pd.DataFrame(
    summaries
)


# ============================================================
# AGGREGATE RESULTS
# ============================================================

overall_accuracy = (
    cases_all["correct"].mean()
)

overall_false_revision = (
    (
        (cases_all["prediction"] == "REVISE") &
        (cases_all["truth"] != "REVISE")
    ).mean()
)

overall_missed_revision = (
    (
        (cases_all["truth"] == "REVISE") &
        (cases_all["prediction"] != "REVISE")
    ).mean()
)

overall_false_recalibration = (
    (
        (cases_all["prediction"] == "RECALIBRATE") &
        (cases_all["truth"] != "RECALIBRATE")
    ).mean()
)

overall_unnecessary_change = (
    (
        (cases_all["truth"] == "KEEP") &
        (cases_all["prediction"] != "KEEP")
    ).mean()
)


# ============================================================
# CLASS RECALL OVER ALL 800 CASES
# ============================================================

class_summary = []

for cls in [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]:

    mask = (
        cases_all["truth"] == cls
    )

    correct = (
        cases_all.loc[
            mask,
            "prediction"
        ] == cls
    ).sum()

    total = mask.sum()

    class_summary.append({
        "class": cls,
        "correct": correct,
        "total": total,
        "recall": correct / total
    })

class_summary_df = pd.DataFrame(
    class_summary
)


# ============================================================
# SAFETY COUNTS
# ============================================================

false_revision_count = (
    (
        (cases_all["prediction"] == "REVISE") &
        (cases_all["truth"] != "REVISE")
    )
).sum()

missed_revision_count = (
    (
        (cases_all["truth"] == "REVISE") &
        (cases_all["prediction"] != "REVISE")
    )
).sum()

quality_violations = (
    (
        (cases_all["quality"] < QUALITY_THRESHOLD) &
        (cases_all["prediction"] != "ABSTAIN")
    )
).sum()


# ============================================================
# PRINT
# ============================================================

print("=" * 80)
print("ETMR — STEP 91: MULTI-SEED INDEPENDENT GENERALIZATION")
print("=" * 80)

print()
print(f"Number of independent seeds = {len(SEEDS)}")
print(f"Cases per seed = {N_PER_CLASS * 4}")
print(f"Total cases = {len(cases_all)}")

print()
print("=" * 80)
print("FROZEN POLICY")
print("=" * 80)

print(
    f"Alpha lower boundary = {ALPHA_LOWER}"
)

print(
    f"Structural threshold = {STRUCTURAL_THRESHOLD}"
)

print(
    f"Quality threshold = {QUALITY_THRESHOLD}"
)

print()
print("=" * 80)
print("PER-SEED PERFORMANCE")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("=" * 80)
print("AGGREGATE PERFORMANCE")
print("=" * 80)

print(
    f"Overall accuracy = "
    f"{overall_accuracy:.4f}"
)

print(
    f"False revision rate = "
    f"{overall_false_revision:.4f}"
)

print(
    f"Missed revision rate = "
    f"{overall_missed_revision:.4f}"
)

print(
    f"False recalibration rate = "
    f"{overall_false_recalibration:.4f}"
)

print(
    f"Unnecessary change rate = "
    f"{overall_unnecessary_change:.4f}"
)

print()
print("=" * 80)
print("AGGREGATE CLASS RECALL")
print("=" * 80)

print(
    class_summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("=" * 80)
print("SAFETY CHECKS")
print("=" * 80)

print(
    f"False structural revisions = "
    f"{false_revision_count}"
)

print(
    f"Missed structural revisions = "
    f"{missed_revision_count}"
)

print(
    f"Quality-gate violations = "
    f"{quality_violations}"
)

if false_revision_count == 0:
    print(
        "PASS: zero false structural revisions."
    )

if missed_revision_count == 0:
    print(
        "PASS: zero missed structural revisions."
    )

if quality_violations == 0:
    print(
        "PASS: zero quality-gate violations."
    )


# ============================================================
# VARIABILITY ACROSS SEEDS
# ============================================================

print()
print("=" * 80)
print("SEED-TO-SEED VARIABILITY")
print("=" * 80)

print(
    f"Mean seed accuracy = "
    f"{summary_df['accuracy'].mean():.4f}"
)

print(
    f"Std seed accuracy = "
    f"{summary_df['accuracy'].std(ddof=1):.4f}"
)

print(
    f"Minimum seed accuracy = "
    f"{summary_df['accuracy'].min():.4f}"
)

print(
    f"Maximum seed accuracy = "
    f"{summary_df['accuracy'].max():.4f}"
)

print()
print("=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

print(
    "This experiment evaluates whether the frozen ETMR "
    "policy remains stable across multiple independently "
    "generated controlled benchmarks."
)

print(
    "No threshold was fitted, modified, or recalibrated "
    "using any of these cases."
)

print(
    "The analysis therefore tests seed-level robustness "
    "rather than threshold optimization."
)

print()
print(
    "IMPORTANT: These are controlled synthetic benchmarks. "
    "High performance here supports frozen-policy robustness "
    "but does not establish universal real-world accuracy."
)


# ============================================================
# SAVE
# ============================================================

cases_all.to_csv(
    "step91_multiseed_cases.csv",
    index=False
)

summary_df.to_csv(
    "step91_multiseed_seed_results.csv",
    index=False
)

class_summary_df.to_csv(
    "step91_multiseed_class_results.csv",
    index=False
)

print()
print("Saved:")
print("step91_multiseed_cases.csv")
print("step91_multiseed_seed_results.csv")
print("step91_multiseed_class_results.csv")

print()
print("=" * 80)
print("STEP 91 COMPLETE")
print("=" * 80)
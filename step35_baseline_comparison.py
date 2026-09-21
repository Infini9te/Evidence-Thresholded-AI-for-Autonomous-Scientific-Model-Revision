import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 35: BASELINE COMPARISON")
print("=" * 80)

# ---------------------------------------------------------
# LOAD UNSEEN BENCHMARK
# ---------------------------------------------------------

df = pd.read_csv("step19_unseen_benchmark.csv")


# ---------------------------------------------------------
# BASELINE 1 — ERROR ONLY
# ---------------------------------------------------------

def error_only(row):

    # Large discrepancy -> REVISE
    if row["MAE"] >= 0.005:
        return "REVISE"

    return "KEEP"


# ---------------------------------------------------------
# BASELINE 2 — PERSISTENCE ONLY
# ---------------------------------------------------------

def persistence_only(row):

    if row["Temporal"] >= 0.70:
        return "RECALIBRATE"

    return "KEEP"


# ---------------------------------------------------------
# BASELINE 3 — CONTEXT ONLY
# ---------------------------------------------------------

def context_only(row):

    if row["Context"] >= 0.70:
        return "REVISE"

    return "KEEP"


# ---------------------------------------------------------
# ETMR — FROZEN POLICY
# ---------------------------------------------------------

QUALITY_THRESHOLD = 0.10
PERSISTENCE_THRESHOLD = 0.70
CONTEXT_THRESHOLD = 0.70


def etmr(row):

    quality = row["Provenance"]
    persistence = row["Temporal"]
    context = row["Context"]

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    if persistence >= PERSISTENCE_THRESHOLD:
        return "RECALIBRATE"

    if (
        context >= CONTEXT_THRESHOLD
        and persistence >= 0.20
    ):
        return "REVISE"

    if (
        persistence < 0.20
        and context < 0.20
    ):
        return "KEEP"

    return "ABSTAIN"


# ---------------------------------------------------------
# APPLY METHODS
# ---------------------------------------------------------

df["Error_Only"] = df.apply(error_only, axis=1)

df["Persistence_Only"] = df.apply(
    persistence_only,
    axis=1
)

df["Context_Only"] = df.apply(
    context_only,
    axis=1
)

df["ETMR"] = df.apply(etmr, axis=1)


# ---------------------------------------------------------
# ACCURACY
# ---------------------------------------------------------

methods = [
    "Error_Only",
    "Persistence_Only",
    "Context_Only",
    "ETMR"
]

results = []

for method in methods:

    correct = (
        df[method] == df["Expected"]
    ).sum()

    accuracy = correct / len(df)

    results.append({
        "Method": method,
        "Correct": correct,
        "Total": len(df),
        "Accuracy": accuracy
    })


results_df = pd.DataFrame(results)


print("\n" + "=" * 80)
print("OVERALL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(index=False)
)


# ---------------------------------------------------------
# PER-CLASS PERFORMANCE
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("PER-CLASS PERFORMANCE")
print("=" * 80)

for method in methods:

    print("\n" + method)

    table = (
        df.groupby("Expected")[method]
        .apply(
            lambda x:
            (x == x.name).mean()
        )
    )

    # Recalculate correctly by expected class
    rows = []

    for expected_class, group in df.groupby("Expected"):

        correct = (
            group[method] == expected_class
        ).sum()

        rows.append({
            "Expected": expected_class,
            "Correct": correct,
            "Cases": len(group),
            "Accuracy": correct / len(group)
        })

    print(
        pd.DataFrame(rows).to_string(index=False)
    )


# ---------------------------------------------------------
# CONFUSION MATRICES
# ---------------------------------------------------------

for method in methods:

    print("\n" + "=" * 80)
    print(f"CONFUSION MATRIX — {method}")
    print("=" * 80)

    confusion = pd.crosstab(
        df["Expected"],
        df[method]
    )

    print(confusion)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv(
    "step35_baseline_comparison_results.csv",
    index=False
)

results_df.to_csv(
    "step35_baseline_summary.csv",
    index=False
)


print("\n" + "=" * 80)
print("STEP 35 COMPLETE")
print("=" * 80)

print("\nSaved:")
print("step35_baseline_comparison_results.csv")
print("step35_baseline_summary.csv")
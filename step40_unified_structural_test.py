import pandas as pd

print("=" * 80)
print("ETMR — STEP 40: UNIFIED STRUCTURAL GENERALIZATION")
print("=" * 80)


# =========================================================
# LOAD RICH UNSEEN MECHANISM RESULTS
# =========================================================

rich = pd.read_csv(
    "step39_rich_etmr_results.csv"
)

print("\nUnseen mechanism cases:")
print(len(rich))


# =========================================================
# LOAD ORIGINAL STRUCTURAL BENCHMARK
# =========================================================

original = pd.read_csv(
    "step14_parameterized_benchmark.csv"
)

original = original[
    original["Expected"] == "REVISE"
].copy()

print("Development structural cases:")
print(len(original))


# =========================================================
# LOAD UNSEEN SEVERITY BENCHMARK
# =========================================================

unseen = pd.read_csv(
    "step19_unseen_benchmark.csv"
)

unseen = unseen[
    unseen["Expected"] == "REVISE"
].copy()

print("Unseen severity cases:")
print(len(unseen))


# =========================================================
# APPLY RICH ETMR TO ORIGINAL / UNSEEN SEVERITY
# =========================================================

QUALITY_THRESHOLD = 0.10
PERSISTENCE_THRESHOLD = 0.70
CONTEXT_THRESHOLD = 0.70
LAG_THRESHOLD = 0.70
INTERACTION_THRESHOLD = 0.60


def rich_etmr(row):

    quality = row["Provenance"]
    persistence = row["Temporal"]
    context = row["Context"]

    # Some old datasets don't contain rich features
    lagged = row.get(
        "Lagged_Rainfall",
        0.0
    )

    interaction = row.get(
        "Interaction",
        0.0
    )

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    if (
        context >= CONTEXT_THRESHOLD
        and
        persistence >= 0.20
    ):
        return "REVISE"

    if lagged >= LAG_THRESHOLD:
        return "REVISE"

    if interaction >= INTERACTION_THRESHOLD:
        return "REVISE"

    if persistence >= PERSISTENCE_THRESHOLD:
        return "RECALIBRATE"

    if (
        persistence < 0.20
        and
        context < 0.20
        and
        lagged < 0.20
        and
        interaction < 0.20
    ):
        return "KEEP"

    return "ABSTAIN"


# =========================================================
# PREPARE OLD DATA
# =========================================================

def prepare_old(df):

    df = df.copy()

    df["Lagged_Rainfall"] = 0.0
    df["Interaction"] = 0.0

    return df


original = prepare_old(original)
unseen = prepare_old(unseen)


# =========================================================
# APPLY
# =========================================================

original["ETMR_Decision"] = original.apply(
    rich_etmr,
    axis=1
)

unseen["ETMR_Decision"] = unseen.apply(
    rich_etmr,
    axis=1
)

rich["ETMR_Decision"] = rich[
    "Rich_ETMR_Decision"
]

original["Correct"] = (
    original["ETMR_Decision"]
    == original["Expected"]
)

unseen["Correct"] = (
    unseen["ETMR_Decision"]
    == unseen["Expected"]
)

rich["Correct"] = (
    rich["ETMR_Decision"]
    == rich["Expected"]
)


# =========================================================
# COMBINE
# =========================================================

columns = [
    "World",
    "Expected",
    "MAE",
    "Temporal",
    "Context",
    "Provenance",
    "ETMR_Decision",
    "Correct"
]

combined = pd.concat(
    [
        original[columns],
        unseen[columns],
        rich[columns]
    ],
    ignore_index=True
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 80)
print("UNIFIED STRUCTURAL RESULTS")
print("=" * 80)

print(
    combined.to_string(index=False)
)


# =========================================================
# ACCURACY
# =========================================================

correct = combined["Correct"].sum()
total = len(combined)

print("\n" + "=" * 80)
print("STRUCTURAL GENERALIZATION")
print("=" * 80)

print(
    f"Correct decisions : {correct} / {total}"
)

print(
    f"Accuracy           : {correct / total:.3f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")

print(
    pd.crosstab(
        combined["Expected"],
        combined["ETMR_Decision"]
    )
)


# =========================================================
# SAVE
# =========================================================

combined.to_csv(
    "step40_unified_structural_results.csv",
    index=False
)

print("\nSaved:")
print("step40_unified_structural_results.csv")

print("\n" + "=" * 80)
print("STEP 40 COMPLETE")
print("=" * 80)
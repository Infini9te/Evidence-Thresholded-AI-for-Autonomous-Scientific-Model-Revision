import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 41: FULL FOUR-WAY BENCHMARK")
print("=" * 80)


# =========================================================
# LOAD BENCHMARKS
# =========================================================

development = pd.read_csv(
    "step14_parameterized_benchmark.csv"
)

unseen = pd.read_csv(
    "step19_unseen_benchmark.csv"
)

rich = pd.read_csv(
    "step39_rich_etmr_results.csv"
)


# =========================================================
# SELECT CASES
# =========================================================

# ---------------------------------------------------------
# KEEP
# ---------------------------------------------------------

keep = development[
    development["Expected"] == "KEEP"
].copy()

# Five development noise cases
keep = keep.head(5)


# ---------------------------------------------------------
# RECALIBRATE
# ---------------------------------------------------------

recalibrate = development[
    development["Expected"] == "RECALIBRATE"
].copy()

recalibrate = recalibrate.head(5)


# ---------------------------------------------------------
# ABSTAIN
# ---------------------------------------------------------

abstain = unseen[
    unseen["Expected"] == "ABSTAIN"
].copy()


# ---------------------------------------------------------
# STRUCTURAL — 15 CASES
# ---------------------------------------------------------

struct_dev = development[
    development["Expected"] == "REVISE"
].copy()

struct_unseen = unseen[
    unseen["Expected"] == "REVISE"
].copy()

struct_rich = rich[
    rich["Expected"] == "REVISE"
].copy()

# Add missing rich evidence columns
struct_dev["Lagged_Rainfall"] = 0.0
struct_dev["Interaction"] = 0.0

struct_unseen["Lagged_Rainfall"] = 0.0
struct_unseen["Interaction"] = 0.0

struct_dev["Source"] = "development_structural"
struct_unseen["Source"] = "unseen_severity"
struct_rich["Source"] = "unseen_mechanism"


# =========================================================
# APPLY FROZEN POLICY
# =========================================================

QUALITY_THRESHOLD = 0.10
PERSISTENCE_THRESHOLD = 0.70
CONTEXT_THRESHOLD = 0.70
LAG_THRESHOLD = 0.70
INTERACTION_THRESHOLD = 0.60


def etmr(row):

    quality = row["Provenance"]

    persistence = row["Temporal"]

    context = row["Context"]

    lagged = row.get(
        "Lagged_Rainfall",
        0.0
    )

    interaction = row.get(
        "Interaction",
        0.0
    )

    # -----------------------------------------------------
    # ABSTAIN
    # -----------------------------------------------------

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # -----------------------------------------------------
    # STRUCTURAL EVIDENCE
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # PARAMETER ERROR
    # -----------------------------------------------------

    if persistence >= PERSISTENCE_THRESHOLD:
        return "RECALIBRATE"

    # -----------------------------------------------------
    # KEEP
    # -----------------------------------------------------

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
# PREPARE STANDARD DATA
# =========================================================

def prepare_basic(df):

    df = df.copy()

    if "Lagged_Rainfall" not in df.columns:
        df["Lagged_Rainfall"] = 0.0

    if "Interaction" not in df.columns:
        df["Interaction"] = 0.0

    if "Source" not in df.columns:
        df["Source"] = "benchmark"

    return df


keep = prepare_basic(keep)
recalibrate = prepare_basic(recalibrate)
abstain = prepare_basic(abstain)
struct_dev = prepare_basic(struct_dev)
struct_unseen = prepare_basic(struct_unseen)
struct_rich = prepare_basic(struct_rich)


# =========================================================
# COMBINE
# =========================================================

combined = pd.concat(
    [
        keep,
        recalibrate,
        abstain,
        struct_dev,
        struct_unseen,
        struct_rich
    ],
    ignore_index=True
)


# =========================================================
# APPLY ETMR
# =========================================================

combined["ETMR_Decision"] = combined.apply(
    etmr,
    axis=1
)

combined["Correct"] = (
    combined["ETMR_Decision"]
    ==
    combined["Expected"]
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 80)
print("FULL BENCHMARK RESULTS")
print("=" * 80)

print(
    combined[
        [
            "World",
            "Expected",
            "Temporal",
            "Context",
            "Lagged_Rainfall",
            "Interaction",
            "Provenance",
            "ETMR_Decision",
            "Correct"
        ]
    ].to_string(index=False)
)


# =========================================================
# OVERALL
# =========================================================

correct = combined["Correct"].sum()
total = len(combined)

print("\n" + "=" * 80)
print("OVERALL PERFORMANCE")
print("=" * 80)

print(
    f"Total cases       : {total}"
)

print(
    f"Correct decisions : {correct}"
)

print(
    f"Accuracy           : {correct / total:.3f}"
)


# =========================================================
# PER-CLASS
# =========================================================

print("\n" + "=" * 80)
print("PER-DECISION PERFORMANCE")
print("=" * 80)

rows = []

for expected, group in combined.groupby("Expected"):

    correct_class = (
        group["Correct"].sum()
    )

    rows.append({
        "Expected": expected,
        "Cases": len(group),
        "Correct": correct_class,
        "Accuracy": correct_class / len(group)
    })

class_results = pd.DataFrame(rows)

print(
    class_results.to_string(index=False)
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

confusion = pd.crosstab(
    combined["Expected"],
    combined["ETMR_Decision"]
)

print(confusion)


# =========================================================
# FALSE REVISION RATE
# =========================================================

false_revisions = (
    (
        combined["ETMR_Decision"] == "REVISE"
    )
    &
    (
        combined["Expected"] != "REVISE"
    )
).sum()

revision_opportunities = (
    combined["Expected"] != "REVISE"
).sum()

false_revision_rate = (
    false_revisions /
    revision_opportunities
)


print("\n" + "=" * 80)
print("REVISION SAFETY")
print("=" * 80)

print(
    f"False revisions : {false_revisions}"
)

print(
    f"False revision rate : "
    f"{false_revision_rate:.3f}"
)


# =========================================================
# MISSED REVISION RATE
# =========================================================

missed_revisions = (
    (
        combined["Expected"] == "REVISE"
    )
    &
    (
        combined["ETMR_Decision"] != "REVISE"
    )
).sum()

total_revisions = (
    combined["Expected"] == "REVISE"
).sum()

missed_revision_rate = (
    missed_revisions /
    total_revisions
)


print(
    f"Missed revisions : {missed_revisions}"
)

print(
    f"Missed revision rate : "
    f"{missed_revision_rate:.3f}"
)


# =========================================================
# SAVE
# =========================================================

combined.to_csv(
    "step41_full_four_way_results.csv",
    index=False
)

class_results.to_csv(
    "step41_four_way_summary.csv",
    index=False
)

print("\nSaved:")
print("step41_full_four_way_results.csv")
print("step41_four_way_summary.csv")

print("\n" + "=" * 80)
print("STEP 41 COMPLETE")
print("=" * 80)
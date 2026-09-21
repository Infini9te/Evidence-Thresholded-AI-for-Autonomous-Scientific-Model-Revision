import pandas as pd
import numpy as np

# ============================================================
# ETMR — STEP 88: COMPONENT ABLATION
# ============================================================

FILE = "step82_final_frozen_evaluation_results.csv"

ALPHA_LOWER = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

df = pd.read_csv(FILE)

truth = df["truth"].to_numpy()

print("=" * 80)
print("ETMR — STEP 88: COMPONENT ABLATION")
print("=" * 80)

print(f"\nLoaded: {FILE}")
print(f"Number of cases = {len(df)}")


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def accuracy(pred):
    return np.mean(pred == truth)


def false_revision(pred):
    return np.mean(
        (truth != "REVISE") &
        (pred == "REVISE")
    )


def missed_revision(pred):
    return np.mean(
        (truth == "REVISE") &
        (pred != "REVISE")
    )


def unnecessary_change(pred):
    return np.mean(
        (truth == "KEEP") &
        np.isin(pred, ["RECALIBRATE", "REVISE"])
    )


def abstain_recall(pred):
    mask = truth == "ABSTAIN"
    return np.mean(pred[mask] == "ABSTAIN")


# ------------------------------------------------------------
# 1. FULL ETMR
# ------------------------------------------------------------

full_etmr = df["predicted"].to_numpy()


# ------------------------------------------------------------
# 2. WITHOUT QUALITY GATE
#
# Structural evidence -> REVISE
# Parameter evidence -> RECALIBRATE
# Otherwise KEEP
#
# Low-quality cases are therefore forced into a scientific
# action rather than ABSTAIN.
# ------------------------------------------------------------

no_quality = np.where(
    df["structural_relative_gain"].to_numpy()
    > STRUCTURAL_THRESHOLD,
    "REVISE",
    np.where(
        df["alpha_case"].to_numpy() < ALPHA_LOWER,
        "RECALIBRATE",
        "KEEP"
    )
)


# ------------------------------------------------------------
# 3. WITHOUT PARAMETER EVIDENCE
#
# Structural evidence -> REVISE
# Otherwise KEEP/ABSTAIN according to quality.
#
# Cannot distinguish KEEP from RECALIBRATE.
# ------------------------------------------------------------

no_parameter = np.where(
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
# 4. WITHOUT STRUCTURAL EVIDENCE
#
# Parameter deviation -> RECALIBRATE
# Otherwise KEEP/ABSTAIN according to quality.
#
# Cannot identify REVISE.
# ------------------------------------------------------------

no_structural = np.where(
    df["quality"].to_numpy() < QUALITY_THRESHOLD,
    "ABSTAIN",
    np.where(
        df["alpha_case"].to_numpy() < ALPHA_LOWER,
        "RECALIBRATE",
        "KEEP"
    )
)


# ------------------------------------------------------------
# 5. WITHOUT BOTH PARAMETER AND STRUCTURAL COMPETITION
#
# Simple quality-gated rule:
# low quality -> ABSTAIN
# high quality -> KEEP
#
# Tests how much the actual evidence discrimination matters.
# ------------------------------------------------------------

quality_only = np.where(
    df["quality"].to_numpy() < QUALITY_THRESHOLD,
    "ABSTAIN",
    "KEEP"
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

methods = {
    "Full ETMR": full_etmr,
    "Without Quality Gate": no_quality,
    "Without Parameter Evidence": no_parameter,
    "Without Structural Evidence": no_structural,
    "Quality Gate Only": quality_only
}

rows = []

for name, pred in methods.items():

    rows.append({
        "method": name,
        "accuracy": accuracy(pred),
        "false_revision_rate": false_revision(pred),
        "missed_revision_rate": missed_revision(pred),
        "unnecessary_change_rate": unnecessary_change(pred),
        "abstain_recall": abstain_recall(pred)
    })

results = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("ABLATION PERFORMANCE")
print("=" * 80)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# Confusion matrices
# ------------------------------------------------------------

labels = [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]

for name, pred in methods.items():

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    cm = pd.crosstab(
        pd.Categorical(
            truth,
            categories=labels
        ),
        pd.Categorical(
            pred,
            categories=labels
        ),
        rownames=["truth"],
        colnames=["predicted"],
        dropna=False
    )

    print(cm)


# ------------------------------------------------------------
# Component contribution
# ------------------------------------------------------------

full_accuracy = accuracy(full_etmr)

print("\n" + "=" * 80)
print("COMPONENT CONTRIBUTION")
print("=" * 80)

for name, pred in methods.items():

    if name == "Full ETMR":
        continue

    loss = full_accuracy - accuracy(pred)

    print(
        f"{name}: accuracy loss relative to full ETMR = "
        f"{loss:.4f}"
    )


# ------------------------------------------------------------
# Scientific interpretation
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

print("""
The ablation experiment evaluates whether ETMR's performance
depends on specific decision components rather than on a generic
classification rule.

Removing the quality gate should primarily damage ABSTAIN
performance because low-quality evidence can no longer be
separated from strong but unreliable discrepancies.

Removing parameter evidence should damage the ability to
distinguish RECALIBRATE from KEEP.

Removing structural evidence should damage the ability to
identify REVISE.

The quality-only rule provides a minimal control that can
identify low-quality cases but cannot determine what scientific
action is justified by the discrepancy.
""")

print("""
IMPORTANT:
This ablation is evaluated on the same frozen controlled
benchmark used in Step 82. It is therefore an attribution
experiment, not an independent generalization test.

The purpose is to establish which ETMR components are necessary
for the observed four-way decision capability.
""")


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results.to_csv(
    "step88_etmr_ablation_results.csv",
    index=False
)

print("\nSaved:")
print("step88_etmr_ablation_results.csv")

print("\n" + "=" * 80)
print("STEP 88 COMPLETE")
print("=" * 80)
import numpy as np
import pandas as pd

# ============================================================
# ETMR — STEP 89: INDEPENDENT FROZEN GENERALIZATION TEST
# ============================================================

SEED = 20260920
N_PER_CLASS = 20

ALPHA_LOWER = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

rng = np.random.default_rng(SEED)

print("=" * 80)
print("ETMR — STEP 89: INDEPENDENT FROZEN GENERALIZATION TEST")
print("=" * 80)

print(f"\nRandom seed = {SEED}")
print(f"Cases per class = {N_PER_CLASS}")
print(f"Total cases = {4 * N_PER_CLASS}")

print("\nFrozen policy:")
print(f"  alpha lower boundary = {ALPHA_LOWER}")
print(f"  structural threshold = {STRUCTURAL_THRESHOLD}")
print(f"  quality threshold = {QUALITY_THRESHOLD}")


# ------------------------------------------------------------
# Generate independent cases
# ------------------------------------------------------------

rows = []

classes = [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]

for truth in classes:

    for i in range(N_PER_CLASS):

        # ----------------------------------------------------
        # KEEP
        # ----------------------------------------------------
        if truth == "KEEP":

            # Deliberately vary around the learned KEEP region.
            alpha = rng.normal(
                loc=1.0405,
                scale=0.00065
            )

            structural_gain = 0.0

            quality = rng.uniform(
                0.82,
                1.00
            )

        # ----------------------------------------------------
        # RECALIBRATE
        # ----------------------------------------------------
        elif truth == "RECALIBRATE":

            # Clearly below KEEP lower boundary but with
            # no structural mechanism.
            alpha = rng.uniform(
                0.90,
                1.030
            )

            structural_gain = 0.0

            quality = rng.uniform(
                0.82,
                1.00
            )

        # ----------------------------------------------------
        # REVISE
        # ----------------------------------------------------
        elif truth == "REVISE":

            # Structural discrepancy is strong.
            alpha = rng.uniform(
                1.70,
                2.60
            )

            structural_gain = rng.uniform(
                0.45,
                0.95
            )

            quality = rng.uniform(
                0.82,
                1.00
            )

        # ----------------------------------------------------
        # ABSTAIN
        # ----------------------------------------------------
        else:

            # Strong apparent discrepancy but insufficient
            # evidence quality.
            alpha = rng.uniform(
                1.70,
                2.60
            )

            structural_gain = rng.uniform(
                0.45,
                0.95
            )

            quality = rng.uniform(
                0.20,
                0.79
            )

        rows.append({
            "case": f"IND_{truth}_{i+1}",
            "truth": truth,
            "alpha_case": alpha,
            "structural_relative_gain": structural_gain,
            "quality": quality
        })


df = pd.DataFrame(rows)


# ------------------------------------------------------------
# Frozen ETMR decision
# ------------------------------------------------------------

predictions = []

for _, row in df.iterrows():

    # Quality gate first
    if row["quality"] < QUALITY_THRESHOLD:

        decision = "ABSTAIN"

    # Structural evidence dominates parameter evidence
    elif row["structural_relative_gain"] > STRUCTURAL_THRESHOLD:

        decision = "REVISE"

    # Parameter deviation
    elif row["alpha_case"] < ALPHA_LOWER:

        decision = "RECALIBRATE"

    # Otherwise retain model
    else:

        decision = "KEEP"

    predictions.append(decision)


df["predicted"] = predictions


# ------------------------------------------------------------
# Performance
# ------------------------------------------------------------

df["correct"] = df["truth"] == df["predicted"]

accuracy = df["correct"].mean()

false_revision = np.mean(
    (df["truth"] != "REVISE") &
    (df["predicted"] == "REVISE")
)

missed_revision = np.mean(
    (df["truth"] == "REVISE") &
    (df["predicted"] != "REVISE")
)

false_recalibration = np.mean(
    (df["truth"] != "RECALIBRATE") &
    (df["predicted"] == "RECALIBRATE")
)

abstain_recall = np.mean(
    df.loc[df["truth"] == "ABSTAIN", "predicted"]
    == "ABSTAIN"
)

abstain_precision = np.mean(
    df.loc[df["predicted"] == "ABSTAIN", "truth"]
    == "ABSTAIN"
)


# ------------------------------------------------------------
# Print performance
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("GENERALIZATION PERFORMANCE")
print("=" * 80)

print(f"\nOverall accuracy = {accuracy:.4f}")
print(f"False revision rate = {false_revision:.4f}")
print(f"Missed revision rate = {missed_revision:.4f}")
print(f"False recalibration rate = {false_recalibration:.4f}")
print(f"ABSTAIN precision = {abstain_precision:.4f}")
print(f"ABSTAIN recall = {abstain_recall:.4f}")


# ------------------------------------------------------------
# Class-level results
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CLASS-LEVEL PERFORMANCE")
print("=" * 80)

class_rows = []

for cls in classes:

    subset = df[df["truth"] == cls]

    class_accuracy = np.mean(
        subset["predicted"] == cls
    )

    class_rows.append({
        "class": cls,
        "correct": int(
            np.sum(subset["predicted"] == cls)
        ),
        "total": len(subset),
        "recall": class_accuracy
    })

class_df = pd.DataFrame(class_rows)

print(
    class_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

labels = [
    "KEEP",
    "RECALIBRATE",
    "REVISE",
    "ABSTAIN"
]

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

cm = pd.crosstab(
    pd.Categorical(
        df["truth"],
        categories=labels
    ),
    pd.Categorical(
        df["predicted"],
        categories=labels
    ),
    rownames=["truth"],
    colnames=["predicted"],
    dropna=False
)

print(cm)


# ------------------------------------------------------------
# Boundary cases
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("BOUNDARY CHECK")
print("=" * 80)

# Count cases close to the alpha boundary
alpha_distance = np.abs(
    df["alpha_case"] - ALPHA_LOWER
)

near_alpha = np.sum(
    alpha_distance < 0.002
)

near_structural = np.sum(
    np.abs(
        df["structural_relative_gain"]
        - STRUCTURAL_THRESHOLD
    ) < 0.05
)

near_quality = np.sum(
    np.abs(
        df["quality"]
        - QUALITY_THRESHOLD
    ) < 0.05
)

print(
    f"Cases near alpha boundary (<0.002) = {near_alpha}"
)

print(
    f"Cases near structural threshold (<0.05) = "
    f"{near_structural}"
)

print(
    f"Cases near quality threshold (<0.05) = "
    f"{near_quality}"
)


# ------------------------------------------------------------
# Scientific safety
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCIENTIFIC SAFETY CHECK")
print("=" * 80)

if false_revision == 0:
    print(
        "PASS: zero false structural revisions."
    )
else:
    print(
        "FAIL: false structural revisions occurred."
    )

if missed_revision == 0:
    print(
        "PASS: zero missed structural revisions."
    )
else:
    print(
        "FAIL: structural revisions were missed."
    )

if abstain_recall == 1.0:
    print(
        "PASS: all low-quality ABSTAIN cases were abstained."
    )
else:
    print(
        "FAIL: some low-quality cases were not abstained."
    )


# ------------------------------------------------------------
# Important methodological statement
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("METHODOLOGICAL NOTE")
print("=" * 80)

print("""
This is an independent frozen-policy stress test.

No ETMR thresholds were fitted using these cases.
No parameters were recalibrated using these cases.
The policy was fixed before generating the benchmark.

The benchmark remains controlled and synthetic, so this result
should be interpreted as evidence of frozen-policy robustness,
not as proof of real-world accuracy.
""")


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_csv(
    "step89_independent_generalization_cases.csv",
    index=False
)

class_df.to_csv(
    "step89_independent_generalization_summary.csv",
    index=False
)

print("\nSaved:")
print("step89_independent_generalization_cases.csv")
print("step89_independent_generalization_summary.csv")

print("\n" + "=" * 80)
print("STEP 89 COMPLETE")
print("=" * 80)
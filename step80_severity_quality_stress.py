import numpy as np
import pandas as pd

print("=" * 80)
print("ETMR — STEP 80: SEVERITY × EVIDENCE-QUALITY STRESS TEST")
print("=" * 80)

# ============================================================
# FROZEN PARAMETERS
# ============================================================

ALPHA_LOW = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

print("\nFrozen rule:")
print(f"KEEP alpha lower boundary : {ALPHA_LOW:.6f}")
print(f"Structural threshold      : {STRUCTURAL_THRESHOLD:.6f}")
print(f"Quality threshold         : {QUALITY_THRESHOLD:.2f}")


# ============================================================
# FROZEN DECISION FUNCTION
# ============================================================

def etmr_decision(alpha_case, structural_gain, quality):

    # Evidence quality has absolute priority.
    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    # Structural evidence sufficient for revision.
    if structural_gain > STRUCTURAL_THRESHOLD:
        return "REVISE"

    # Parameter deviation sufficient for recalibration.
    if alpha_case < ALPHA_LOW:
        return "RECALIBRATE"

    # Otherwise retain the model.
    return "KEEP"


# ============================================================
# STRESS GRID
# ============================================================

structural_strengths = [
    0.05,
    0.15,
    0.25,
    0.40,
    0.55,
    0.70,
    0.85,
    0.95
]

qualities = [
    1.00,
    0.90,
    0.80,
    0.70,
    0.50,
    0.30
]


# ============================================================
# GENERATE CASES
# ============================================================

cases = []

for quality in qualities:

    for strength in structural_strengths:

        prediction = etmr_decision(
            alpha_case=1.0407,
            structural_gain=strength,
            quality=quality
        )

        cases.append({
            "structural_strength": strength,
            "quality": quality,
            "predicted": prediction
        })

df = pd.DataFrame(cases)


# ============================================================
# PRINT FULL GRID
# ============================================================

print("\n" + "=" * 80)
print("DECISION GRID")
print("=" * 80)

grid = df.pivot(
    index="quality",
    columns="structural_strength",
    values="predicted"
)

print(grid.to_string())


# ============================================================
# STRUCTURAL MONOTONICITY
# ============================================================

print("\n" + "=" * 80)
print("STRUCTURAL SEVERITY RESPONSE")
print("=" * 80)

for quality in qualities:

    subset = df[df["quality"] == quality].sort_values(
        "structural_strength"
    )

    print(f"\nQuality = {quality:.2f}")

    for _, row in subset.iterrows():
        print(
            f"strength={row['structural_strength']:.2f} "
            f"-> {row['predicted']}"
        )


# ============================================================
# CHECK EXPECTED TRANSITIONS
# ============================================================

print("\n" + "=" * 80)
print("MONOTONICITY CHECKS")
print("=" * 80)

# At sufficiently high quality, decisions should move from
# non-revision toward revision as structural evidence increases.

high_quality = df[df["quality"] >= QUALITY_THRESHOLD]

high_quality_revision_boundary = (
    high_quality[
        high_quality["structural_strength"] > STRUCTURAL_THRESHOLD
    ]["predicted"] == "REVISE"
).all()

print(
    "High-quality strong evidence -> REVISE: "
    f"{high_quality_revision_boundary}"
)

# At low quality, EVERYTHING should abstain regardless
# of structural discrepancy.

low_quality = df[df["quality"] < QUALITY_THRESHOLD]

low_quality_all_abstain = (
    low_quality["predicted"] == "ABSTAIN"
).all()

print(
    "Low-quality evidence -> ABSTAIN regardless of severity: "
    f"{low_quality_all_abstain}"
)


# ============================================================
# COUNT DECISIONS
# ============================================================

print("\n" + "=" * 80)
print("DECISION COUNTS")
print("=" * 80)

print(
    df["predicted"]
    .value_counts()
    .sort_index()
)


# ============================================================
# BOUNDARY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("STRUCTURAL THRESHOLD ANALYSIS")
print("=" * 80)

print(
    f"Frozen structural threshold = "
    f"{STRUCTURAL_THRESHOLD:.6f}"
)

below = [
    x for x in structural_strengths
    if x <= STRUCTURAL_THRESHOLD
]

above = [
    x for x in structural_strengths
    if x > STRUCTURAL_THRESHOLD
]

print(f"Strengths at/below threshold: {below}")
print(f"Strengths above threshold:    {above}")


# ============================================================
# SAVE
# ============================================================

output_file = "step80_severity_quality_stress_results.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 80)
print("STEP 80 COMPLETE")
print("=" * 80)
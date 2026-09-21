import numpy as np
import pandas as pd

# ============================================================
# ETMR — STEP 90
# BOUNDARY ROBUSTNESS ANALYSIS
# ============================================================

SEED = 20260920
N_PER_CLASS = 20

# ============================================================
# FROZEN POLICY — DO NOT CHANGE
# ============================================================

ALPHA_LOWER = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

rng = np.random.default_rng(SEED)

rows = []

# ============================================================
# RECONSTRUCT EXACT STEP 89 BENCHMARK
# ============================================================

# KEEP
for i in range(N_PER_CLASS):

    alpha = rng.normal(
        loc=1.0405,
        scale=0.00065
    )

    rows.append({
        "case_id": f"IND_KEEP_{i+1:02d}",
        "truth": "KEEP",
        "alpha": alpha,
        "structural_gain": 0.0,
        "quality": rng.uniform(0.82, 1.00)
    })


# RECALIBRATE
for i in range(N_PER_CLASS):

    alpha = rng.uniform(
        0.90,
        1.03
    )

    rows.append({
        "case_id": f"IND_RECALIBRATE_{i+1:02d}",
        "truth": "RECALIBRATE",
        "alpha": alpha,
        "structural_gain": 0.0,
        "quality": rng.uniform(0.82, 1.00)
    })


# REVISE
for i in range(N_PER_CLASS):

    alpha = rng.uniform(
        1.70,
        2.60
    )

    rows.append({
        "case_id": f"IND_REVISE_{i+1:02d}",
        "truth": "REVISE",
        "alpha": alpha,
        "structural_gain": rng.uniform(0.45, 0.95),
        "quality": rng.uniform(0.82, 1.00)
    })


# ABSTAIN
for i in range(N_PER_CLASS):

    alpha = rng.uniform(
        1.70,
        2.60
    )

    rows.append({
        "case_id": f"IND_ABSTAIN_{i+1:02d}",
        "truth": "ABSTAIN",
        "alpha": alpha,
        "structural_gain": rng.uniform(0.45, 0.95),
        "quality": rng.uniform(0.20, 0.79)
    })


df = pd.DataFrame(rows)

# ============================================================
# FROZEN ETMR DECISION
# ============================================================

def etmr_decision(row):

    if row["quality"] < QUALITY_THRESHOLD:
        return "ABSTAIN"

    if row["structural_gain"] > STRUCTURAL_THRESHOLD:
        return "REVISE"

    if row["alpha"] < ALPHA_LOWER:
        return "RECALIBRATE"

    return "KEEP"


df["prediction"] = df.apply(
    etmr_decision,
    axis=1
)

df["correct"] = (
    df["truth"] == df["prediction"]
)

# ============================================================
# BOUNDARY DISTANCES
# ============================================================

df["alpha_distance"] = np.abs(
    df["alpha"] - ALPHA_LOWER
)

df["structural_distance"] = np.abs(
    df["structural_gain"] - STRUCTURAL_THRESHOLD
)

df["quality_distance"] = np.abs(
    df["quality"] - QUALITY_THRESHOLD
)

# ============================================================
# ERROR ANALYSIS
# ============================================================

errors = df[
    ~df["correct"]
].copy()

keep = df[
    df["truth"] == "KEEP"
].copy()

recal = df[
    df["truth"] == "RECALIBRATE"
].copy()

keep_errors = keep[
    ~keep["correct"]
].copy()

recal_errors = recal[
    ~recal["correct"]
].copy()

# ============================================================
# ALPHA DISTRIBUTION
# ============================================================

keep_min = keep["alpha"].min()
keep_max = keep["alpha"].max()

recal_min = recal["alpha"].min()
recal_max = recal["alpha"].max()

overlap_min = max(
    keep_min,
    recal_min
)

overlap_max = min(
    keep_max,
    recal_max
)

overlap_exists = (
    overlap_min <= overlap_max
)

overlap_width = (
    overlap_max - overlap_min
    if overlap_exists
    else 0.0
)

# ============================================================
# ERROR DISTANCE
# ============================================================

if len(errors) > 0:

    mean_error_distance = (
        errors["alpha_distance"].mean()
    )

    min_error_distance = (
        errors["alpha_distance"].min()
    )

    max_error_distance = (
        errors["alpha_distance"].max()
    )

else:

    mean_error_distance = np.nan
    min_error_distance = np.nan
    max_error_distance = np.nan

# ============================================================
# BOUNDARY WINDOWS
# ============================================================

windows = [
    0.0005,
    0.001,
    0.002,
    0.005,
    0.010
]

boundary_rows = []

for w in windows:

    near = (
        df["alpha_distance"] < w
    )

    total_near = near.sum()

    errors_near = (
        near &
        (~df["correct"])
    ).sum()

    boundary_rows.append({
        "alpha_window": w,
        "cases_near_boundary": total_near,
        "errors_near_boundary": errors_near,
        "error_rate_near_boundary":
            (
                errors_near / total_near
                if total_near > 0
                else np.nan
            )
    })

boundary_summary = pd.DataFrame(
    boundary_rows
)

# ============================================================
# SAFETY CHECKS
# ============================================================

false_revision = (
    (df["prediction"] == "REVISE") &
    (df["truth"] != "REVISE")
)

missed_revision = (
    (df["truth"] == "REVISE") &
    (df["prediction"] != "REVISE")
)

quality_violations = (
    (df["quality"] < QUALITY_THRESHOLD) &
    (df["prediction"] != "ABSTAIN")
)

# ============================================================
# PRINT
# ============================================================

print("=" * 80)
print("ETMR — STEP 90: BOUNDARY ROBUSTNESS ANALYSIS")
print("=" * 80)

print()
print(f"Independent seed = {SEED}")
print(f"Cases per class = {N_PER_CLASS}")
print(f"Total cases = {len(df)}")

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
print("OVERALL ERROR ANALYSIS")
print("=" * 80)

print(
    f"Total errors = {len(errors)}"
)

print(
    f"Error rate = {(~df['correct']).mean():.4f}"
)

if len(errors) > 0:

    print()
    print("Error types:")

    print(
        errors[
            ["truth", "prediction"]
        ].value_counts().to_string()
    )

print()
print("=" * 80)
print("ALPHA DISTRIBUTION")
print("=" * 80)

print(
    f"KEEP alpha min = {keep_min:.6f}"
)

print(
    f"KEEP alpha max = {keep_max:.6f}"
)

print(
    f"RECALIBRATE alpha min = {recal_min:.6f}"
)

print(
    f"RECALIBRATE alpha max = {recal_max:.6f}"
)

print()
print(
    f"Alpha overlap exists = {overlap_exists}"
)

print(
    f"Overlap width = {overlap_width:.6f}"
)

print()
print("=" * 80)
print("KEEP BOUNDARY ANALYSIS")
print("=" * 80)

print(
    f"KEEP total = {len(keep)}"
)

print(
    f"KEEP correct = {keep['correct'].sum()}"
)

print(
    f"KEEP errors = {len(keep_errors)}"
)

print(
    f"KEEP recall = "
    f"{keep['correct'].mean():.4f}"
)

if len(keep_errors) > 0:

    print()
    print("KEEP errors:")

    print(
        keep_errors[
            [
                "case_id",
                "alpha",
                "alpha_distance",
                "prediction"
            ]
        ].to_string(index=False)
    )

print()
print("=" * 80)
print("RECALIBRATE ANALYSIS")
print("=" * 80)

print(
    f"RECALIBRATE total = {len(recal)}"
)

print(
    f"RECALIBRATE correct = {recal['correct'].sum()}"
)

print(
    f"RECALIBRATE errors = {len(recal_errors)}"
)

print(
    f"RECALIBRATE recall = "
    f"{recal['correct'].mean():.4f}"
)

if len(recal_errors) > 0:

    print()
    print("RECALIBRATE errors:")

    print(
        recal_errors[
            [
                "case_id",
                "alpha",
                "alpha_distance",
                "prediction"
            ]
        ].to_string(index=False)
    )

print()
print("=" * 80)
print("ERROR DISTANCE FROM ALPHA BOUNDARY")
print("=" * 80)

print(
    f"Mean error distance = "
    f"{mean_error_distance:.6f}"
)

print(
    f"Minimum error distance = "
    f"{min_error_distance:.6f}"
)

print(
    f"Maximum error distance = "
    f"{max_error_distance:.6f}"
)

print()
print("=" * 80)
print("BOUNDARY WINDOW ANALYSIS")
print("=" * 80)

print(
    boundary_summary.to_string(
        index=False
    )
)

print()
print("=" * 80)
print("STRUCTURAL / SAFETY CHECK")
print("=" * 80)

print(
    f"False structural revisions = "
    f"{false_revision.sum()}"
)

print(
    f"Missed structural revisions = "
    f"{missed_revision.sum()}"
)

print(
    f"Quality-gate violations = "
    f"{quality_violations.sum()}"
)

if false_revision.sum() == 0:
    print(
        "PASS: zero false structural revisions."
    )

if missed_revision.sum() == 0:
    print(
        "PASS: zero missed structural revisions."
    )

if quality_violations.sum() == 0:
    print(
        "PASS: zero quality-gate violations."
    )

print()
print("=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

if len(keep_errors) > 0:

    print(
        "The observed independent errors occur in the "
        "KEEP class and are associated with the frozen "
        "parameter boundary."
    )

    print(
        "This indicates sensitivity of the KEEP versus "
        "RECALIBRATE decision near the parameter boundary."
    )

else:

    print(
        "No KEEP boundary errors were observed."
    )

print()
print(
    "No thresholds were modified."
)

print(
    "This analysis localizes boundary sensitivity rather "
    "than optimizing the frozen policy."
)

print()
print(
    "IMPORTANT: This remains a controlled synthetic "
    "generalization analysis and is not proof of universal "
    "real-world accuracy."
)

# ============================================================
# SAVE
# ============================================================

errors.to_csv(
    "step90_boundary_errors.csv",
    index=False
)

boundary_summary.to_csv(
    "step90_boundary_summary.csv",
    index=False
)

df.to_csv(
    "step90_boundary_full_cases.csv",
    index=False
)

print()
print("Saved:")
print("step90_boundary_errors.csv")
print("step90_boundary_summary.csv")
print("step90_boundary_full_cases.csv")

print()
print("=" * 80)
print("STEP 90 COMPLETE")
print("=" * 80)
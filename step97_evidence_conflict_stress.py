import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 97: EVIDENCE CONFLICT STRESS TEST")
print("=" * 80)

# ---------------------------------------------------------
# FROZEN POLICY — DO NOT MODIFY
# ---------------------------------------------------------

ALPHA_LOWER = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80

print("\nFrozen thresholds:")
print(f"  alpha lower          = {ALPHA_LOWER}")
print(f"  structural threshold = {STRUCTURAL_THRESHOLD}")
print(f"  quality threshold    = {QUALITY_THRESHOLD}")

# ---------------------------------------------------------
# FROZEN DECISION FUNCTION
# ---------------------------------------------------------

def etmr_decision(alpha, structural_gain, quality):

    if quality < QUALITY_THRESHOLD:
        return "ABSTAIN"

    if structural_gain > STRUCTURAL_THRESHOLD:
        return "REVISE"

    if alpha < ALPHA_LOWER:
        return "RECALIBRATE"

    return "KEEP"


# ---------------------------------------------------------
# CONFLICT CASES
# ---------------------------------------------------------

cases = [

    # -----------------------------------------------------
    # 1. KEEP
    # -----------------------------------------------------
    {
        "case_id": "CONFLICT_KEEP_01",
        "truth": "KEEP",
        "alpha": 1.0405,
        "structural_gain": 0.10,
        "quality": 0.95,
        "scenario": "Weak parameter and structural evidence, high quality"
    },

    {
        "case_id": "CONFLICT_KEEP_02",
        "truth": "KEEP",
        "alpha": 1.0408,
        "structural_gain": 0.38,
        "quality": 0.90,
        "scenario": "Structural evidence just below revision threshold"
    },

    # -----------------------------------------------------
    # 2. RECALIBRATE
    # -----------------------------------------------------
    {
        "case_id": "CONFLICT_RECAL_01",
        "truth": "RECALIBRATE",
        "alpha": 0.95,
        "structural_gain": 0.10,
        "quality": 0.95,
        "scenario": "Strong parameter evidence, weak structural evidence"
    },

    {
        "case_id": "CONFLICT_RECAL_02",
        "truth": "RECALIBRATE",
        "alpha": 1.00,
        "structural_gain": 0.38,
        "quality": 0.90,
        "scenario": "Parameter evidence strong, structural evidence just below threshold"
    },

    # -----------------------------------------------------
    # 3. REVISE
    # -----------------------------------------------------
    {
        "case_id": "CONFLICT_REVISE_01",
        "truth": "REVISE",
        "alpha": 1.0405,
        "structural_gain": 0.80,
        "quality": 0.95,
        "scenario": "Strong structural evidence, adequate parameter fit"
    },

    {
        "case_id": "CONFLICT_REVISE_02",
        "truth": "REVISE",
        "alpha": 0.95,
        "structural_gain": 0.80,
        "quality": 0.95,
        "scenario": "Strong parameter and structural evidence"
    },

    # -----------------------------------------------------
    # 4. ABSTAIN
    # -----------------------------------------------------
    {
        "case_id": "CONFLICT_ABSTAIN_01",
        "truth": "ABSTAIN",
        "alpha": 0.95,
        "structural_gain": 0.80,
        "quality": 0.60,
        "scenario": "Strong parameter and structural evidence, poor quality"
    },

    {
        "case_id": "CONFLICT_ABSTAIN_02",
        "truth": "ABSTAIN",
        "alpha": 1.0405,
        "structural_gain": 0.80,
        "quality": 0.75,
        "scenario": "Strong structural evidence, insufficient quality"
    },

    {
        "case_id": "CONFLICT_ABSTAIN_03",
        "truth": "ABSTAIN",
        "alpha": 0.95,
        "structural_gain": 0.40,
        "quality": 0.79,
        "scenario": "Both evidence dimensions suggest change, but quality fails"
    },

    # -----------------------------------------------------
    # 5. BOUNDARY CASES
    # -----------------------------------------------------
    {
        "case_id": "BOUNDARY_ALPHA_01",
        "truth": "KEEP",
        "alpha": ALPHA_LOWER + 0.000001,
        "structural_gain": 0.00,
        "quality": 1.00,
        "scenario": "Alpha infinitesimally above frozen boundary"
    },

    {
        "case_id": "BOUNDARY_ALPHA_02",
        "truth": "RECALIBRATE",
        "alpha": ALPHA_LOWER - 0.000001,
        "structural_gain": 0.00,
        "quality": 1.00,
        "scenario": "Alpha infinitesimally below frozen boundary"
    },

    {
        "case_id": "BOUNDARY_STRUCTURAL_01",
        "truth": "KEEP",
        "alpha": 1.0405,
        "structural_gain": STRUCTURAL_THRESHOLD - 0.000001,
        "quality": 1.00,
        "scenario": "Structural evidence infinitesimally below boundary"
    },

    {
        "case_id": "BOUNDARY_STRUCTURAL_02",
        "truth": "REVISE",
        "alpha": 1.0405,
        "structural_gain": STRUCTURAL_THRESHOLD + 0.000001,
        "quality": 1.00,
        "scenario": "Structural evidence infinitesimally above boundary"
    },

    {
        "case_id": "BOUNDARY_QUALITY_01",
        "truth": "ABSTAIN",
        "alpha": 0.95,
        "structural_gain": 0.80,
        "quality": QUALITY_THRESHOLD - 0.000001,
        "scenario": "Quality infinitesimally below boundary"
    },

    {
        "case_id": "BOUNDARY_QUALITY_02",
        "truth": "REVISE",
        "alpha": 1.0405,
        "structural_gain": 0.80,
        "quality": QUALITY_THRESHOLD + 0.000001,
        "scenario": "Quality infinitesimally above boundary"
    }
]

df = pd.DataFrame(cases)

# ---------------------------------------------------------
# RUN FROZEN POLICY
# ---------------------------------------------------------

df["prediction"] = df.apply(
    lambda row: etmr_decision(
        row["alpha"],
        row["structural_gain"],
        row["quality"]
    ),
    axis=1
)

df["correct"] = df["truth"] == df["prediction"]

# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CASE-LEVEL RESULTS")
print("=" * 80)

print(
    df[
        [
            "case_id",
            "truth",
            "prediction",
            "alpha",
            "structural_gain",
            "quality",
            "correct",
            "scenario"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

accuracy = df["correct"].mean()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total cases = {len(df)}")
print(f"Accuracy    = {accuracy:.4f}")

print("\nConfusion matrix:")
print(
    pd.crosstab(
        df["truth"],
        df["prediction"]
    )
)

# ---------------------------------------------------------
# SAFETY METRICS
# ---------------------------------------------------------

false_revision = (
    (df["prediction"] == "REVISE") &
    (df["truth"] != "REVISE")
).mean()

missed_revision = (
    (df["truth"] == "REVISE") &
    (df["prediction"] != "REVISE")
).mean()

quality_violations = (
    (df["quality"] < QUALITY_THRESHOLD) &
    (df["prediction"] != "ABSTAIN")
).sum()

print("\n" + "=" * 80)
print("SAFETY")
print("=" * 80)

print(f"False revision rate = {false_revision:.4f}")
print(f"Missed revision rate = {missed_revision:.4f}")
print(f"Quality-gate violations = {quality_violations}")

# ---------------------------------------------------------
# CONFLICT-SPECIFIC CHECKS
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CONFLICT CHECKS")
print("=" * 80)

checks = {

    "Low quality overrides strong structural evidence":
        (
            df.loc[
                df["case_id"].str.contains("ABSTAIN"),
                "prediction"
            ] == "ABSTAIN"
        ).all(),

    "Strong structural evidence produces REVISE":
        (
            df.loc[
                df["case_id"].str.contains("REVISE"),
                "prediction"
            ] == "REVISE"
        ).all(),

    "Parameter evidence produces RECALIBRATE when structural evidence is weak":
        (
            df.loc[
                df["case_id"].str.contains("RECAL"),
                "prediction"
            ] == "RECALIBRATE"
        ).all(),

    "Alpha boundary changes decision":
        (
            df.loc[
                df["case_id"].isin(
                    ["BOUNDARY_ALPHA_01", "BOUNDARY_ALPHA_02"]
                ),
                "prediction"
            ].tolist()
            == ["KEEP", "RECALIBRATE"]
        ),

    "Structural boundary changes decision":
        (
            df.loc[
                df["case_id"].isin(
                    ["BOUNDARY_STRUCTURAL_01", "BOUNDARY_STRUCTURAL_02"]
                ),
                "prediction"
            ].tolist()
            == ["KEEP", "REVISE"]
        ),

    "Quality boundary changes decision":
        (
            df.loc[
                df["case_id"].isin(
                    ["BOUNDARY_QUALITY_01", "BOUNDARY_QUALITY_02"]
                ),
                "prediction"
            ].tolist()
            == ["ABSTAIN", "REVISE"]
        )
}

for name, result in checks.items():
    print(f"{name}: {'PASS' if result else 'FAIL'}")

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv(
    "step97_evidence_conflict_stress.csv",
    index=False
)

print("\nSaved:")
print("  step97_evidence_conflict_stress.csv")

print("\n" + "=" * 80)
print("STEP 97 COMPLETE")
print("=" * 80)
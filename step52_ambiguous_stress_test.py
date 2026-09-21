import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 52: AMBIGUOUS / OVERLAPPING EVIDENCE STRESS TEST")
print("=" * 80)

rng = np.random.default_rng(20260910)

cases = []

# ============================================================
# 1. KEEP / RE-CALIBRATE BOUNDARY
# ============================================================

for i in range(10):

    cases.append({
        "World": f"BOUNDARY_PARAMETER_{i+1:02d}",
        "Expected": "RECALIBRATE",
        "MAE": rng.uniform(0.002, 0.008),
        "Temporal": rng.uniform(0.65, 0.78),
        "Context": rng.uniform(0.15, 0.35),
        "Lagged_Rainfall": rng.uniform(0.05, 0.25),
        "Interaction": rng.uniform(0.05, 0.25),
        "Q_evidence": 1.0
    })


# ============================================================
# 2. PARAMETER / STRUCTURAL BOUNDARY
# ============================================================

for i in range(10):

    cases.append({
        "World": f"BOUNDARY_STRUCTURE_{i+1:02d}",
        "Expected": "REVISE",
        "MAE": rng.uniform(0.005, 0.025),
        "Temporal": rng.uniform(0.45, 0.75),
        "Context": rng.uniform(0.45, 0.75),
        "Lagged_Rainfall": rng.uniform(0.20, 0.65),
        "Interaction": rng.uniform(0.20, 0.65),
        "Q_evidence": 1.0
    })


# ============================================================
# 3. STRONG STRUCTURE BUT LOW EVIDENCE QUALITY
# ============================================================

for i, q in enumerate(
    np.linspace(0.05, 0.95, 10)
):

    cases.append({
        "World": f"BOUNDARY_QUALITY_{i+1:02d}",
        "Expected": "ABSTAIN" if q < 0.80 else "REVISE",
        "MAE": rng.uniform(0.015, 0.040),
        "Temporal": rng.uniform(0.30, 0.60),
        "Context": rng.uniform(0.80, 0.98),
        "Lagged_Rainfall": rng.uniform(0.10, 0.30),
        "Interaction": rng.uniform(0.10, 0.30),
        "Q_evidence": q
    })


# ============================================================
# 4. STRONG LAGGED EVIDENCE BUT WEAK CONTEMPORANEOUS CONTEXT
# ============================================================

for i in range(10):

    cases.append({
        "World": f"BOUNDARY_LAGGED_{i+1:02d}",
        "Expected": "REVISE",
        "MAE": rng.uniform(0.005, 0.030),
        "Temporal": rng.uniform(0.20, 0.60),
        "Context": rng.uniform(0.00, 0.30),
        "Lagged_Rainfall": rng.uniform(0.70, 0.98),
        "Interaction": rng.uniform(0.00, 0.25),
        "Q_evidence": 1.0
    })


# ============================================================
# 5. CONFLICTING EVIDENCE
# ============================================================

for i in range(10):

    cases.append({
        "World": f"CONFLICTING_{i+1:02d}",
        "Expected": "ABSTAIN",
        "MAE": rng.uniform(0.005, 0.030),
        "Temporal": rng.uniform(0.65, 0.90),
        "Context": rng.uniform(0.60, 0.90),
        "Lagged_Rainfall": rng.uniform(0.40, 0.70),
        "Interaction": rng.uniform(0.40, 0.70),
        "Q_evidence": rng.uniform(0.40, 0.75)
    })


# ============================================================
# Create dataframe
# ============================================================

df = pd.DataFrame(cases)

print("\nTotal cases:", len(df))

print("\nClass distribution:")
print(df["Expected"].value_counts())

print("\nEvidence ranges:")
print(
    df[
        [
            "Temporal",
            "Context",
            "Lagged_Rainfall",
            "Interaction",
            "Q_evidence"
        ]
    ].describe()
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    "step52_ambiguous_stress_test.csv",
    index=False
)

print("\nSaved:")
print("step52_ambiguous_stress_test.csv")

print("\n" + "=" * 80)
print("STEP 52 COMPLETE")
print("=" * 80)
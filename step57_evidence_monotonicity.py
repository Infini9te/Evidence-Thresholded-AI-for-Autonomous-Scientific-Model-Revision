import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 57: EVIDENCE MONOTONICITY TEST")
print("=" * 80)

# ============================================================
# FROZEN POLICY FOR THIS TEST
# ============================================================

QUALITY_THRESHOLD = 0.80
PARAMETER_THRESHOLD = 0.30
STRUCTURAL_THRESHOLD = 0.30


def etmr_decision(q, parameter_gain, structural_gain):

    if q < QUALITY_THRESHOLD:
        return "ABSTAIN"

    if (
        parameter_gain >= PARAMETER_THRESHOLD
        and structural_gain < STRUCTURAL_THRESHOLD
    ):
        return "RECALIBRATE"

    if structural_gain >= STRUCTURAL_THRESHOLD:
        return "REVISE"

    if (
        parameter_gain < 0.30
        and structural_gain < 0.30
    ):
        return "KEEP"

    return "ABSTAIN"


# ============================================================
# PART A — STRUCTURAL EVIDENCE VS QUALITY
# ============================================================

print("\n" + "-" * 80)
print("PART A: STRUCTURAL EVIDENCE VS OBSERVATION QUALITY")
print("-" * 80)

qualities = np.linspace(1.00, 0.05, 20)

structural_results = []

for q in qualities:

    parameter_gain = 0.15
    structural_gain = 0.85

    decision = etmr_decision(
        q,
        parameter_gain,
        structural_gain
    )

    structural_results.append({
        "Experiment": "STRUCTURAL_QUALITY",
        "Q_evidence": q,
        "Parameter_Gain": parameter_gain,
        "Structural_Gain": structural_gain,
        "Decision": decision
    })

structural_df = pd.DataFrame(structural_results)

print(
    structural_df.to_string(index=False)
)


# ============================================================
# PART B — PARAMETER EVIDENCE VS STRENGTH
# ============================================================

print("\n" + "-" * 80)
print("PART B: PARAMETER EVIDENCE STRENGTH")
print("-" * 80)

parameter_strengths = np.linspace(0.00, 0.90, 19)

parameter_results = []

for p in parameter_strengths:

    q = 1.0
    structural_gain = 0.10

    decision = etmr_decision(
        q,
        p,
        structural_gain
    )

    parameter_results.append({
        "Experiment": "PARAMETER_STRENGTH",
        "Q_evidence": q,
        "Parameter_Gain": p,
        "Structural_Gain": structural_gain,
        "Decision": decision
    })

parameter_df = pd.DataFrame(parameter_results)

print(
    parameter_df.to_string(index=False)
)


# ============================================================
# PART C — STRUCTURAL EVIDENCE STRENGTH
# ============================================================

print("\n" + "-" * 80)
print("PART C: STRUCTURAL EVIDENCE STRENGTH")
print("-" * 80)

structural_strengths = np.linspace(0.00, 0.90, 19)

structural_strength_results = []

for s in structural_strengths:

    q = 1.0
    parameter_gain = 0.10

    decision = etmr_decision(
        q,
        parameter_gain,
        s
    )

    structural_strength_results.append({
        "Experiment": "STRUCTURAL_STRENGTH",
        "Q_evidence": q,
        "Parameter_Gain": parameter_gain,
        "Structural_Gain": s,
        "Decision": decision
    })

structural_strength_df = pd.DataFrame(
    structural_strength_results
)

print(
    structural_strength_df.to_string(index=False)
)


# ============================================================
# PART D — CHECK MONOTONICITY
# ============================================================

print("\n" + "-" * 80)
print("MONOTONICITY CHECKS")
print("-" * 80)


# Structural quality:
# As quality decreases, REVISE should eventually become ABSTAIN.

quality_decisions = structural_df["Decision"].tolist()

revise_to_abstain = (
    "REVISE" in quality_decisions
    and "ABSTAIN" in quality_decisions
)

print(
    "Structural evidence: "
    "REVISE → ABSTAIN transition:",
    revise_to_abstain
)


# Parameter:
# As parameter evidence increases,
# decision should eventually become RECALIBRATE.

parameter_decisions = parameter_df["Decision"].tolist()

parameter_transition = (
    "RECALIBRATE" in parameter_decisions
)

print(
    "Parameter evidence reaches RECALIBRATE:",
    parameter_transition
)


# Structural:
# As structural evidence increases,
# decision should eventually become REVISE.

structural_decisions = structural_strength_df[
    "Decision"
].tolist()

structural_transition = (
    "REVISE" in structural_decisions
)

print(
    "Structural evidence reaches REVISE:",
    structural_transition
)


# ============================================================
# SAVE
# ============================================================

all_results = pd.concat(
    [
        structural_df,
        parameter_df,
        structural_strength_df
    ],
    ignore_index=True
)

all_results.to_csv(
    "step57_evidence_monotonicity_results.csv",
    index=False
)

print("\nSaved:")
print("step57_evidence_monotonicity_results.csv")

print("\n" + "=" * 80)
print("STEP 57 COMPLETE")
print("=" * 80)
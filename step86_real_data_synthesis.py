import pandas as pd
import numpy as np

# ============================================================
# ETMR — STEP 86: REAL-DATA SCIENTIFIC SYNTHESIS
# ============================================================

print("=" * 80)
print("ETMR — STEP 86: REAL-DATA SCIENTIFIC SYNTHESIS")
print("=" * 80)

# ------------------------------------------------------------
# Real-data results from Steps 58–63
# These values are frozen from the completed analyses.
# ------------------------------------------------------------

periods = [
    {
        "period": 2022,
        "parameter_mae_gain": 0.00003626,
        "parameter_rmse_gain": -0.00003940,
        "parameter_supported": False,
        "structural_mae_gain": 0.00000665,
        "structural_rmse_gain": 0.00001228,
        "structural_reduction": 0.00063001,
        "structural_supported": False,
        "decision_directional": "REVISE",
        "final_temporal_status": "NOT_SUPPORTED"
    },
    {
        "period": 2023,
        "parameter_mae_gain": 0.00000320,
        "parameter_rmse_gain": -0.00001705,
        "parameter_supported": False,
        "structural_mae_gain": 0.00000205,
        "structural_rmse_gain": -0.00000681,
        "structural_reduction": -0.00059453,
        "structural_supported": False,
        "decision_directional": "KEEP",
        "final_temporal_status": "NOT_SUPPORTED"
    },
    {
        "period": 2024,
        "parameter_mae_gain": 0.00000636,
        "parameter_rmse_gain": 0.00005308,
        "parameter_supported": True,
        "structural_mae_gain": 0.00001279,
        "structural_rmse_gain": 0.00003094,
        "structural_reduction": -0.00312603,
        "structural_supported": False,
        "decision_directional": "RECALIBRATE",
        "final_temporal_status": "SUPPORTED"
    }
]

df = pd.DataFrame(periods)

# ------------------------------------------------------------
# Multi-period reproducibility
# ------------------------------------------------------------

structural_supported_periods = int(
    df["structural_supported"].sum()
)

parameter_supported_periods = int(
    df["parameter_supported"].sum()
)

n_periods = len(df)

structural_reproducibility = (
    structural_supported_periods / n_periods
)

parameter_reproducibility = (
    parameter_supported_periods / n_periods
)

# ------------------------------------------------------------
# Frozen temporal decision
# Minimum 2 supported periods required for a change
# ------------------------------------------------------------

MIN_SUPPORTED_PERIODS = 2

if structural_supported_periods >= MIN_SUPPORTED_PERIODS:
    temporal_structural_decision = "REVISE"
else:
    temporal_structural_decision = "NOT_SUPPORTED"

if parameter_supported_periods >= MIN_SUPPORTED_PERIODS:
    temporal_parameter_decision = "RECALIBRATE"
else:
    temporal_parameter_decision = "NOT_SUPPORTED"

if temporal_structural_decision == "REVISE":
    final_decision = "REVISE"
elif temporal_parameter_decision == "RECALIBRATE":
    final_decision = "RECALIBRATE"
else:
    final_decision = "KEEP"

# ------------------------------------------------------------
# Print period-level table
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PERIOD-LEVEL REAL-DATA EVIDENCE")
print("=" * 80)

print(
    df[
        [
            "period",
            "parameter_mae_gain",
            "parameter_rmse_gain",
            "parameter_supported",
            "structural_mae_gain",
            "structural_rmse_gain",
            "structural_reduction",
            "structural_supported",
            "decision_directional",
            "final_temporal_status"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)

# ------------------------------------------------------------
# Reproducibility summary
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TEMPORAL REPRODUCIBILITY")
print("=" * 80)

print(
    f"\nStructural support = "
    f"{structural_supported_periods}/{n_periods} "
    f"({structural_reproducibility:.3f})"
)

print(
    f"Parameter support = "
    f"{parameter_supported_periods}/{n_periods} "
    f"({parameter_reproducibility:.3f})"
)

print(
    f"\nMinimum supported periods required = "
    f"{MIN_SUPPORTED_PERIODS}"
)

print(
    f"Structural temporal decision = "
    f"{temporal_structural_decision}"
)

print(
    f"Parameter temporal decision = "
    f"{temporal_parameter_decision}"
)

print(
    f"\nFINAL REAL-DATA ETMR DECISION = "
    f"{final_decision}"
)

# ------------------------------------------------------------
# Scientific interpretation
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

print("""
The real-data analysis does not provide reproducible evidence
that the current irrigation model requires structural revision.

A structural signal appears directionally in 2022, but the
structural counterfactual evidence is not statistically supported
across the multi-period bootstrap analysis.

Parameter correction receives stronger support in 2024, but this
support is not reproduced across the other evaluation periods.

Because neither structural nor parameter evidence reaches the
frozen minimum reproducibility requirement of two supported
periods, ETMR selects KEEP.

This is an intentional scientific outcome: the framework avoids
changing the scientific model when apparent discrepancies are
transient or insufficiently reproducible.
""")

# ------------------------------------------------------------
# Scientific claim boundaries
# ------------------------------------------------------------

print("=" * 80)
print("CLAIM BOUNDARIES")
print("=" * 80)

print("""
SUPPORTED CLAIM:
ETMR can integrate parameter, structural, evidence-quality,
and temporal-reproducibility evidence into a four-way
scientific model-change decision.

SUPPORTED CONTROLLED RESULT:
On the frozen 60-case blind benchmark, ETMR achieved 100%
accuracy and outperformed the strongest simple baseline
(75%) by 25 percentage points.

REAL-DATA RESULT:
For the present irrigation dataset, ETMR selects KEEP because
neither structural revision nor parameter recalibration is
supported reproducibly across the evaluated periods.

NOT SUPPORTED:
These experiments do not establish that the irrigation model
is universally correct, nor do they establish 100% real-world
decision accuracy.

The real-data result should therefore be presented as evidence
of conservative model-change governance, not proof of model
truth.
""")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_csv(
    "step86_real_data_period_results.csv",
    index=False
)

summary = pd.DataFrame([{
    "structural_supported_periods": structural_supported_periods,
    "total_periods": n_periods,
    "structural_reproducibility": structural_reproducibility,
    "parameter_supported_periods": parameter_supported_periods,
    "parameter_reproducibility": parameter_reproducibility,
    "minimum_supported_periods": MIN_SUPPORTED_PERIODS,
    "temporal_structural_decision": temporal_structural_decision,
    "temporal_parameter_decision": temporal_parameter_decision,
    "final_real_data_decision": final_decision
}])

summary.to_csv(
    "step86_real_data_synthesis_summary.csv",
    index=False
)

print("\nSaved:")
print("step86_real_data_period_results.csv")
print("step86_real_data_synthesis_summary.csv")

print("\n" + "=" * 80)
print("STEP 86 COMPLETE")
print("=" * 80)
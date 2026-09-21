import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 99: FINAL REAL-DATA SCIENTIFIC EVALUATION")
print("=" * 80)

# ------------------------------------------------------------------
# Frozen temporal evidence from Steps 62–63
# ------------------------------------------------------------------

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
    }
]

df = pd.DataFrame(periods)

# ------------------------------------------------------------------
# Frozen temporal reproducibility requirement
# ------------------------------------------------------------------

MIN_SUPPORTED_PERIODS = 2
TOTAL_PERIODS = len(df)

parameter_support_count = int(df["parameter_supported"].sum())
structural_support_count = int(df["structural_supported"].sum())

parameter_reproducible = parameter_support_count >= MIN_SUPPORTED_PERIODS
structural_reproducible = structural_support_count >= MIN_SUPPORTED_PERIODS

# ------------------------------------------------------------------
# Final conservative scientific decision
# ------------------------------------------------------------------

if structural_reproducible:
    final_decision = "REVISE"

elif parameter_reproducible:
    final_decision = "RECALIBRATE"

else:
    final_decision = "KEEP"

df["final_temporal_policy_decision"] = final_decision

# ------------------------------------------------------------------
# Scientific interpretation
# ------------------------------------------------------------------

if final_decision == "KEEP":
    interpretation = (
        "Neither structural revision nor parameter recalibration has "
        "sufficient reproducible support across the evaluated periods. "
        "The conservative ETMR decision is KEEP."
    )

elif final_decision == "RECALIBRATE":
    interpretation = (
        "Parameter evidence is reproducible across the required number "
        "of temporal periods, while structural evidence is not. "
        "ETMR therefore selects RECALIBRATE."
    )

else:
    interpretation = (
        "Structural evidence is reproducible across the required number "
        "of temporal periods. ETMR therefore selects REVISE."
    )

# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("TEMPORAL EVIDENCE")
print("=" * 80)

print(df.to_string(index=False))

print("\n" + "=" * 80)
print("REPRODUCIBILITY")
print("=" * 80)

print(f"Total evaluated periods          = {TOTAL_PERIODS}")
print(f"Minimum supported periods       = {MIN_SUPPORTED_PERIODS}")
print(f"Parameter-supported periods     = {parameter_support_count}/{TOTAL_PERIODS}")
print(f"Structural-supported periods    = {structural_support_count}/{TOTAL_PERIODS}")
print(f"Parameter reproducible          = {parameter_reproducible}")
print(f"Structural reproducible         = {structural_reproducible}")

print("\n" + "=" * 80)
print("FINAL REAL-DATA ETMR DECISION")
print("=" * 80)

print(f"FINAL DECISION = {final_decision}")

print("\nScientific interpretation:")
print(interpretation)

# ------------------------------------------------------------------
# Claim boundary
# ------------------------------------------------------------------

claim_boundary = (
    "This result demonstrates the application of ETMR to the available "
    "irrigation dataset and its temporal evidence. It does not establish "
    "universal real-world accuracy or prove that the scientific model "
    "is universally adequate."
)

print("\n" + "=" * 80)
print("CLAIM BOUNDARY")
print("=" * 80)

print(claim_boundary)

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

df.to_csv(
    "step99_final_real_data_period_results.csv",
    index=False
)

pd.DataFrame([
    {
        "metric": "parameter_supported_periods",
        "value": f"{parameter_support_count}/{TOTAL_PERIODS}"
    },
    {
        "metric": "structural_supported_periods",
        "value": f"{structural_support_count}/{TOTAL_PERIODS}"
    },
    {
        "metric": "minimum_required_periods",
        "value": MIN_SUPPORTED_PERIODS
    },
    {
        "metric": "final_decision",
        "value": final_decision
    },
    {
        "metric": "interpretation",
        "value": interpretation
    }
]).to_csv(
    "step99_final_real_data_summary.csv",
    index=False
)

print("\nSaved:")
print("  step99_final_real_data_period_results.csv")
print("  step99_final_real_data_summary.csv")

print("\n" + "=" * 80)
print("STEP 99 COMPLETE")
print("=" * 80)
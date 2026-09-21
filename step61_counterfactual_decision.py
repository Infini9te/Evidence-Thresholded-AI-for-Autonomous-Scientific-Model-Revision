import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 61: UNIFIED COUNTERFACTUAL DECISION")
print("=" * 80)

# ============================================================
# 1. LOAD RESULTS FROM STEPS 59 AND 60
# ============================================================

param = pd.read_csv(
    "step59_bootstrap_parameter_counterfactual_results.csv"
)

struct = pd.read_csv(
    "step60_bootstrap_structural_counterfactual_results.csv"
)

param = dict(
    zip(
        param["Metric"],
        param["Value"]
    )
)

struct = dict(
    zip(
        struct["Metric"],
        struct["Value"]
    )
)

# ============================================================
# 2. EXTRACT PARAMETER EVIDENCE
# ============================================================

parameter_rmse_gain = param[
    "Observed_RMSE_Gain"
]

parameter_rmse_ci_lower = param[
    "RMSE_CI_Lower"
]

parameter_rmse_ci_upper = param[
    "RMSE_CI_Upper"
]

parameter_p_rmse = param[
    "P_RMSE_Gain_Positive"
]

parameter_mae_gain = param[
    "Observed_MAE_Gain"
]

parameter_mae_ci_lower = param[
    "MAE_CI_Lower"
]

parameter_mae_ci_upper = param[
    "MAE_CI_Upper"
]

parameter_p_mae = param[
    "P_MAE_Gain_Positive"
]

alpha = param["alpha"]

# ============================================================
# 3. EXTRACT STRUCTURAL EVIDENCE
# ============================================================

struct_rmse_gain = struct[
    "Observed_RMSE_Gain"
]

struct_rmse_ci_lower = struct[
    "RMSE_CI_Lower"
]

struct_rmse_ci_upper = struct[
    "RMSE_CI_Upper"
]

struct_p_rmse = struct[
    "P_RMSE_Gain_Positive"
]

struct_mae_gain = struct[
    "Observed_MAE_Gain"
]

struct_mae_ci_lower = struct[
    "MAE_CI_Lower"
]

struct_mae_ci_upper = struct[
    "MAE_CI_Upper"
]

struct_p_mae = struct[
    "P_MAE_Gain_Positive"
]

structure_reduction = struct[
    "Observed_Structure_Reduction"
]

structure_ci_lower = struct[
    "Structure_CI_Lower"
]

structure_ci_upper = struct[
    "Structure_CI_Upper"
]

structure_p = struct[
    "P_Structure_Reduction_Positive"
]

# ============================================================
# 4. DEFINE ROBUST EVIDENCE
# ============================================================

parameter_rmse_robust = (
    parameter_rmse_ci_lower > 0
    and parameter_p_rmse >= 0.95
)

parameter_mae_robust = (
    parameter_mae_ci_lower > 0
    and parameter_p_mae >= 0.95
)

struct_rmse_robust = (
    struct_rmse_ci_lower > 0
    and struct_p_rmse >= 0.95
)

struct_mae_robust = (
    struct_mae_ci_lower > 0
    and struct_p_mae >= 0.95
)

structural_pattern_robust = (
    structure_ci_lower > 0
    and structure_p >= 0.95
)

# ============================================================
# 5. PARAMETER COUNTERFACTUAL SCORE
# ============================================================

parameter_support = 0

if parameter_rmse_robust:
    parameter_support += 1

if parameter_mae_robust:
    parameter_support += 1

# ============================================================
# 6. STRUCTURAL COUNTERFACTUAL SCORE
# ============================================================

structural_support = 0

if struct_rmse_robust:
    structural_support += 1

if struct_mae_robust:
    structural_support += 1

if structural_pattern_robust:
    structural_support += 1

# ============================================================
# 7. PRINT PARAMETER EVIDENCE
# ============================================================

print("\n" + "-" * 80)
print("PARAMETER COUNTERFACTUAL")
print("-" * 80)

print(
    f"Alpha correction              = {alpha:.8f}"
)

print(
    f"RMSE gain                     = {parameter_rmse_gain:.8f}"
)

print(
    f"RMSE 95% CI                   = "
    f"[{parameter_rmse_ci_lower:.8f}, "
    f"{parameter_rmse_ci_upper:.8f}]"
)

print(
    f"P(RMSE gain > 0)              = "
    f"{parameter_p_rmse:.4f}"
)

print(
    f"MAE gain                      = {parameter_mae_gain:.8f}"
)

print(
    f"MAE 95% CI                    = "
    f"[{parameter_mae_ci_lower:.8f}, "
    f"{parameter_mae_ci_upper:.8f}]"
)

print(
    f"P(MAE gain > 0)               = "
    f"{parameter_p_mae:.4f}"
)

print(
    f"Robust RMSE evidence          = "
    f"{parameter_rmse_robust}"
)

print(
    f"Robust MAE evidence           = "
    f"{parameter_mae_robust}"
)

print(
    f"Parameter support count       = "
    f"{parameter_support}/2"
)

# ============================================================
# 8. PRINT STRUCTURAL EVIDENCE
# ============================================================

print("\n" + "-" * 80)
print("STRUCTURAL COUNTERFACTUAL")
print("-" * 80)

print(
    f"RMSE gain                     = {struct_rmse_gain:.8f}"
)

print(
    f"RMSE 95% CI                   = "
    f"[{struct_rmse_ci_lower:.8f}, "
    f"{struct_rmse_ci_upper:.8f}]"
)

print(
    f"P(RMSE gain > 0)              = "
    f"{struct_p_rmse:.4f}"
)

print(
    f"MAE gain                      = {struct_mae_gain:.8f}"
)

print(
    f"MAE 95% CI                    = "
    f"[{struct_mae_ci_lower:.8f}, "
    f"{struct_mae_ci_upper:.8f}]"
)

print(
    f"P(MAE gain > 0)               = "
    f"{struct_p_mae:.4f}"
)

print(
    f"Residual structure reduction  = "
    f"{structure_reduction:.8f}"
)

print(
    f"Structure reduction 95% CI    = "
    f"[{structure_ci_lower:.8f}, "
    f"{structure_ci_upper:.8f}]"
)

print(
    f"P(structure reduction > 0)    = "
    f"{structure_p:.4f}"
)

print(
    f"Robust RMSE evidence          = "
    f"{struct_rmse_robust}"
)

print(
    f"Robust MAE evidence           = "
    f"{struct_mae_robust}"
)

print(
    f"Robust structural removal     = "
    f"{structural_pattern_robust}"
)

print(
    f"Structural support count      = "
    f"{structural_support}/3"
)

# ============================================================
# 9. DECISION LOGIC
# ============================================================

print("\n" + "-" * 80)
print("ETMR COUNTERFACTUAL DECISION")
print("-" * 80)

# Strong structural evidence requires:
# 1. robust predictive improvement
# 2. robust removal of residual structure

if (
    struct_rmse_robust
    and structural_pattern_robust
):
    decision = "REVISE"

# Strong parameter evidence requires:
# robust predictive improvement in both metrics

elif (
    parameter_rmse_robust
    and parameter_mae_robust
):
    decision = "RECALIBRATE"

else:
    decision = "KEEP"

print(
    "\nFinal ETMR decision:",
    decision
)

# ============================================================
# 10. SCIENTIFIC INTERPRETATION
# ============================================================

if decision == "REVISE":

    interpretation = (
        "Structural counterfactual provides robust predictive "
        "improvement and removes the targeted residual structure."
    )

elif decision == "RECALIBRATE":

    interpretation = (
        "Parameter recalibration provides robust predictive "
        "improvement without requiring structural revision."
    )

else:

    interpretation = (
        "Neither parameter recalibration nor the tested structural "
        "counterfactual provides sufficient evidence for model change."
    )

print(
    "\nScientific interpretation:"
)

print(
    interpretation
)

# ============================================================
# 11. SAVE
# ============================================================

summary = pd.DataFrame({
    "Evidence": [
        "Parameter_RMSE_Robust",
        "Parameter_MAE_Robust",
        "Parameter_Support_Count",
        "Structural_RMSE_Robust",
        "Structural_MAE_Robust",
        "Structural_Removal_Robust",
        "Structural_Support_Count",
        "Alpha",
        "Final_ETMR_Decision"
    ],
    "Value": [
        parameter_rmse_robust,
        parameter_mae_robust,
        parameter_support,
        struct_rmse_robust,
        struct_mae_robust,
        structural_pattern_robust,
        structural_support,
        alpha,
        decision
    ]
})

summary.to_csv(
    "step61_counterfactual_decision.csv",
    index=False
)

print("\nSaved:")
print("step61_counterfactual_decision.csv")

print("\n" + "=" * 80)
print("STEP 61 COMPLETE")
print("=" * 80)
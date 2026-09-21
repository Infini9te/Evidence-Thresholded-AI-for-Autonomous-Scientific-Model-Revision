import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 27: FORMAL EVIDENCE VECTOR")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD STEP 26 RESULTS
# ---------------------------------------------------------

df = pd.read_csv(
    "step26_model_evidence_summary.csv"
)

row = df.iloc[0]

# ---------------------------------------------------------
# 2. EXTRACT EVIDENCE
# ---------------------------------------------------------

mae_improvement = row["MAE_Improvement"]
mae_ci_lower = row["MAE_CI_Lower"]
mae_ci_upper = row["MAE_CI_Upper"]

rmse_improvement = row["RMSE_Improvement"]
rmse_ci_lower = row["RMSE_CI_Lower"]
rmse_ci_upper = row["RMSE_CI_Upper"]

structure_reduction = row["Structure_Reduction"]
structure_ci_lower = row["Structure_CI_Lower"]
structure_ci_upper = row["Structure_CI_Upper"]

p_mae = row["P_M1_Better"]
p_rmse = row["P_RMSE_Improvement"]
p_structure = row["P_Structure_Reduction"]

# ---------------------------------------------------------
# 3. DEFINE EVIDENCE CONDITIONS
# ---------------------------------------------------------

# Prediction evidence:
# We require the MAE improvement CI to exclude zero.

prediction_supported = (
    mae_ci_lower > 0
)

# Stronger predictive evidence:
# RMSE improvement must also exclude zero.

rmse_supported = (
    rmse_ci_lower > 0
)

# Structural evidence:
# The residual-structure reduction must exclude zero.

structure_supported = (
    structure_ci_lower > 0
)

# Stability:
# Require high bootstrap probability.

stability_supported = (
    p_mae >= 0.95
    and
    p_structure >= 0.95
)

# ---------------------------------------------------------
# 4. EVIDENCE VECTOR
# ---------------------------------------------------------

evidence_vector = {
    "Prediction_Evidence": int(prediction_supported),
    "RMSE_Evidence": int(rmse_supported),
    "Structural_Evidence": int(structure_supported),
    "Stability_Evidence": int(stability_supported)
}

print("\nEVIDENCE VECTOR")
print("-" * 80)

for key, value in evidence_vector.items():
    print(f"{key}: {value}")

# ---------------------------------------------------------
# 5. ETMR DECISION
# ---------------------------------------------------------

if (
    prediction_supported
    and
    structure_supported
    and
    stability_supported
):

    decision = "REVISE"

elif (
    not prediction_supported
    and
    not structure_supported
):

    decision = "KEEP"

else:

    decision = "INSUFFICIENT_EVIDENCE"

# ---------------------------------------------------------
# 6. REPORT
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ETMR DECISION")
print("=" * 80)

print(
    "Prediction CI:",
    f"[{mae_ci_lower:.9f}, {mae_ci_upper:.9f}]"
)

print(
    "Structural CI:",
    f"[{structure_ci_lower:.6f}, {structure_ci_upper:.6f}]"
)

print(
    "P(M1 better by MAE):",
    f"{p_mae:.4f}"
)

print(
    "P(structure reduction):",
    f"{p_structure:.4f}"
)

print(
    "\nFINAL ETMR DECISION:",
    decision
)

# ---------------------------------------------------------
# 7. SAVE
# ---------------------------------------------------------

output = pd.DataFrame([
    {
        **evidence_vector,
        "MAE_Improvement": mae_improvement,
        "MAE_CI_Lower": mae_ci_lower,
        "MAE_CI_Upper": mae_ci_upper,
        "RMSE_Improvement": rmse_improvement,
        "RMSE_CI_Lower": rmse_ci_lower,
        "RMSE_CI_Upper": rmse_ci_upper,
        "Structure_Reduction": structure_reduction,
        "Structure_CI_Lower": structure_ci_lower,
        "Structure_CI_Upper": structure_ci_upper,
        "P_MAE_Better": p_mae,
        "P_RMSE_Better": p_rmse,
        "P_Structure_Reduction": p_structure,
        "ETMR_Decision": decision
    }
])

output.to_csv(
    "step27_etmr_evidence_vector.csv",
    index=False
)

print("\nSaved:")
print("step27_etmr_evidence_vector.csv")

print("\n" + "=" * 80)
print("STEP 27 COMPLETE")
print("=" * 80)
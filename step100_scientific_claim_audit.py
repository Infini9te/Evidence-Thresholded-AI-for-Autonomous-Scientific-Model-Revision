import pandas as pd

print("=" * 80)
print("ETMR — STEP 100: FINAL SCIENTIFIC CLAIM AUDIT")
print("=" * 80)

claims = [

    {
        "claim_id": "C1",
        "claim_type": "CORE_METHOD",
        "claim": "ETMR frames model modification as an evidence-thresholded "
                 "scientific decision among KEEP, RECALIBRATE, REVISE, and ABSTAIN.",
        "evidence": "Step 98 formal algorithm specification",
        "strength": "SUPPORTED",
        "allowed": "YES"
    },

    {
        "claim_id": "C2",
        "claim_type": "COMPONENT_ROLE",
        "claim": "Quality, parameter, and structural evidence provide distinct "
                 "decision functions within ETMR.",
        "evidence": "Step 88 component ablation",
        "strength": "SUPPORTED",
        "allowed": "YES"
    },

    {
        "claim_id": "C3",
        "claim_type": "BASELINE_ADVANTAGE",
        "claim": "ETMR outperforms the structural-plus-quality baseline on the "
                 "800-case independent controlled benchmark.",
        "evidence": "Step 95",
        "strength": "SUPPORTED",
        "allowed": "YES"
    },

    {
        "claim_id": "C4",
        "claim_type": "STATISTICAL_ROBUSTNESS",
        "claim": "The 800-case ETMR accuracy advantage is statistically robust "
                 "within the controlled benchmark.",
        "evidence": "Steps 93–95",
        "strength": "SUPPORTED_WITH_SCOPE",
        "allowed": "YES"
    },

    {
        "claim_id": "C5",
        "claim_type": "SAFETY",
        "claim": "ETMR produced zero false structural revisions and zero missed "
                 "structural revisions on the evaluated controlled benchmarks.",
        "evidence": "Steps 83, 88, 94, 95",
        "strength": "SUPPORTED_WITH_SCOPE",
        "allowed": "YES"
    },

    {
        "claim_id": "C6",
        "claim_type": "CONFLICT_HANDLING",
        "claim": "ETMR correctly handled deliberately constructed conflicts between "
                 "parameter evidence, structural evidence, and evidence quality.",
        "evidence": "Step 97",
        "strength": "SUPPORTED",
        "allowed": "YES"
    },

    {
        "claim_id": "C7",
        "claim_type": "REAL_DATA",
        "claim": "On the irrigation dataset, ETMR selected KEEP because neither "
                 "structural nor parameter evidence was temporally reproducible.",
        "evidence": "Step 99",
        "strength": "SUPPORTED",
        "allowed": "YES"
    },

    {
        "claim_id": "C8",
        "claim_type": "GENERALIZATION",
        "claim": "ETMR is universally accurate for real-world scientific model revision.",
        "evidence": "No real-world validation establishing universality",
        "strength": "NOT_SUPPORTED",
        "allowed": "NO"
    },

    {
        "claim_id": "C9",
        "claim_type": "REAL_WORLD_ACCURACY",
        "claim": "ETMR achieves 98.12% accuracy on real irrigation data.",
        "evidence": "98.12% comes from the controlled 800-case benchmark",
        "strength": "NOT_SUPPORTED",
        "allowed": "NO"
    },

    {
        "claim_id": "C10",
        "claim_type": "CAUSALITY",
        "claim": "ETMR proves that rainfall is causally responsible for model discrepancy.",
        "evidence": "Current experiments do not establish causal identification",
        "strength": "NOT_SUPPORTED",
        "allowed": "NO"
    }
]

df = pd.DataFrame(claims)

print("\n" + "=" * 80)
print("CLAIM AUDIT")
print("=" * 80)

print(df.to_string(index=False))

# ---------------------------------------------------------
# Strongest contributions
# ---------------------------------------------------------

contributions = pd.DataFrame([
    {
        "rank": 1,
        "contribution":
            "Evidence-thresholded four-way model revision decision",
        "why_important":
            "Moves beyond prediction error toward deciding whether a "
            "scientific model should be kept, recalibrated, revised, "
            "or left undecided."
    },
    {
        "rank": 2,
        "contribution":
            "Competing parameter-versus-structural counterfactual evidence",
        "why_important":
            "Separates parameter inadequacy from genuine structural "
            "inadequacy rather than treating residual magnitude as "
            "automatic evidence for revision."
    },
    {
        "rank": 3,
        "contribution":
            "Evidence quality and temporal reproducibility as safeguards",
        "why_important":
            "Prevents apparently strong but unreliable or non-reproducible "
            "evidence from triggering unjustified model changes."
    }
])

print("\n" + "=" * 80)
print("THREE PRIMARY SCIENTIFIC CONTRIBUTIONS")
print("=" * 80)

print(contributions.to_string(index=False))

# ---------------------------------------------------------
# Final quantitative evidence
# ---------------------------------------------------------

quantitative = pd.DataFrame([
    {
        "result": "Independent controlled benchmark",
        "value": "800 cases",
        "interpretation": "Controlled validation"
    },
    {
        "result": "ETMR accuracy",
        "value": "98.12%",
        "interpretation": "Controlled benchmark only"
    },
    {
        "result": "Strongest baseline accuracy",
        "value": "75.00%",
        "interpretation": "Structural + Quality baseline"
    },
    {
        "result": "Accuracy advantage",
        "value": "+23.12 percentage points",
        "interpretation": "Controlled benchmark only"
    },
    {
        "result": "Bootstrap 95% CI",
        "value": "[19.88, 26.37] percentage points",
        "interpretation": "Controlled benchmark only"
    },
    {
        "result": "False structural revisions",
        "value": "0",
        "interpretation": "Evaluated controlled benchmarks"
    },
    {
        "result": "Missed structural revisions",
        "value": "0",
        "interpretation": "Evaluated controlled benchmarks"
    },
    {
        "result": "Real-data structural support",
        "value": "0/3 periods",
        "interpretation": "No reproducible structural evidence"
    },
    {
        "result": "Real-data parameter support",
        "value": "1/3 periods",
        "interpretation": "Insufficient for reproducible recalibration"
    },
    {
        "result": "Final real-data decision",
        "value": "KEEP",
        "interpretation": "Conservative ETMR decision"
    }
])

print("\n" + "=" * 80)
print("FINAL QUANTITATIVE EVIDENCE")
print("=" * 80)

print(quantitative.to_string(index=False))

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    "step100_scientific_claim_audit.csv",
    index=False
)

contributions.to_csv(
    "step100_primary_contributions.csv",
    index=False
)

quantitative.to_csv(
    "step100_final_quantitative_evidence.csv",
    index=False
)

print("\nSaved:")
print("  step100_scientific_claim_audit.csv")
print("  step100_primary_contributions.csv")
print("  step100_final_quantitative_evidence.csv")

print("\n" + "=" * 80)
print("STEP 100 COMPLETE")
print("=" * 80)
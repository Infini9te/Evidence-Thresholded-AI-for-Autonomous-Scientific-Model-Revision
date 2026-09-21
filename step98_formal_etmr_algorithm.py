import pandas as pd

print("=" * 80)
print("ETMR — STEP 98: FORMAL ALGORITHM SPECIFICATION")
print("=" * 80)

# ---------------------------------------------------------
# FROZEN POLICY
# ---------------------------------------------------------

ALPHA_LOWER = 1.039534
STRUCTURAL_THRESHOLD = 0.388612
QUALITY_THRESHOLD = 0.80
MIN_SUPPORTED_PERIODS = 2
TOTAL_PERIODS = 3

# ---------------------------------------------------------
# FORMAL ETMR SPECIFICATION
# ---------------------------------------------------------

specification = {

    "name":
        "Evidence-Thresholded Model Revision (ETMR)",

    "objective":
        "Determine whether scientific model behavior should be "
        "kept unchanged, recalibrated, structurally revised, "
        "or left undecided because evidence is insufficient.",

    "input":
        "Observed system data, scientific baseline model, "
        "parameter counterfactual, structural counterfactual, "
        "measurement/provenance quality, and temporal evidence.",

    "output":
        "KEEP, RECALIBRATE, REVISE, or ABSTAIN",

    "decision_hierarchy":
        "1. Evidence quality gate; "
        "2. structural counterfactual evidence; "
        "3. case-specific parameter adequacy; "
        "4. KEEP if neither change is sufficiently supported.",

    "quality_gate":
        "If evidence quality < 0.80, return ABSTAIN.",

    "structural_rule":
        "If evidence quality >= 0.80 and structural relative gain "
        "> 0.388612, return REVISE.",

    "parameter_rule":
        "If evidence quality >= 0.80, structural evidence is below "
        "revision threshold, and case-specific parameter estimate "
        "< 1.039534, return RECALIBRATE.",

    "keep_rule":
        "If evidence quality >= 0.80, structural evidence is below "
        "revision threshold, and case-specific parameter estimate "
        ">= 1.039534, return KEEP.",

    "temporal_rule":
        "A change should be accepted as reproducibly supported only "
        "when the corresponding evidence is supported in at least "
        "2 of 3 evaluated temporal periods.",

    "parameter_evidence_role":
        "Tests whether parameter adjustment can explain the observed "
        "discrepancy without changing model structure.",

    "structural_evidence_role":
        "Tests whether a structural/mechanistic counterfactual "
        "provides materially stronger explanatory adequacy.",

    "quality_evidence_role":
        "Prevents strong apparent discrepancies from being interpreted "
        "as scientifically actionable when observations are unreliable.",

    "temporal_evidence_role":
        "Prevents a model change from being justified by an isolated "
        "period-specific improvement.",

    "decision_philosophy":
        "Model modification requires evidence sufficient to discriminate "
        "among competing explanations of model discrepancy."
}

# ---------------------------------------------------------
# PSEUDOCODE
# ---------------------------------------------------------

pseudocode = [
    "INPUT observed data D and baseline scientific model M0",
    "ESTIMATE baseline model performance and residual structure",
    "ESTIMATE evidence quality Q from provenance and observation reliability",
    "IF Q < 0.80: RETURN ABSTAIN",
    "ESTIMATE case-specific parameter adequacy A",
    "ESTIMATE structural counterfactual advantage G",
    "IF G > 0.388612: candidate = REVISE",
    "ELSE IF A < 1.039534: candidate = RECALIBRATE",
    "ELSE: candidate = KEEP",
    "EVALUATE candidate evidence across temporal periods",
    "IF required temporal reproducibility is not satisfied:",
    "    RETURN KEEP or ABSTAIN according to evidence sufficiency",
    "RETURN final ETMR decision"
]

# ---------------------------------------------------------
# SAVE SPECIFICATION
# ---------------------------------------------------------

rows = []

for key, value in specification.items():
    rows.append({
        "component": key,
        "formal_definition": value
    })

spec_df = pd.DataFrame(rows)

spec_df.to_csv(
    "step98_formal_etmr_specification.csv",
    index=False
)

pd.DataFrame({
    "step": range(1, len(pseudocode) + 1),
    "pseudocode": pseudocode
}).to_csv(
    "step98_etmr_pseudocode.csv",
    index=False
)

# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("FROZEN ETMR PARAMETERS")
print("=" * 80)

print(f"Alpha lower threshold       = {ALPHA_LOWER}")
print(f"Structural threshold        = {STRUCTURAL_THRESHOLD}")
print(f"Quality threshold           = {QUALITY_THRESHOLD}")
print(f"Minimum supported periods   = {MIN_SUPPORTED_PERIODS}")
print(f"Temporal periods evaluated  = {TOTAL_PERIODS}")

print("\n" + "=" * 80)
print("FORMAL ETMR SPECIFICATION")
print("=" * 80)

for key, value in specification.items():
    print(f"\n{key.upper()}:")
    print(value)

print("\n" + "=" * 80)
print("ETMR PSEUDOCODE")
print("=" * 80)

for i, line in enumerate(pseudocode, 1):
    print(f"{i:02d}. {line}")

print("\nSaved:")
print("  step98_formal_etmr_specification.csv")
print("  step98_etmr_pseudocode.csv")

print("\n" + "=" * 80)
print("STEP 98 COMPLETE")
print("=" * 80)
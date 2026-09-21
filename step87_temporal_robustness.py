import pandas as pd

# ============================================================
# ETMR — STEP 87: TEMPORAL ROBUSTNESS ANALYSIS
# ============================================================

print("=" * 80)
print("ETMR — STEP 87: TEMPORAL ROBUSTNESS ANALYSIS")
print("=" * 80)

# Frozen period-level support from Step 63
# Do NOT modify these values.

data = pd.DataFrame([
    {
        "period": 2022,
        "parameter_supported": False,
        "structural_supported": False
    },
    {
        "period": 2023,
        "parameter_supported": False,
        "structural_supported": False
    },
    {
        "period": 2024,
        "parameter_supported": True,
        "structural_supported": False
    }
])

n_periods = len(data)

parameter_supported = int(
    data["parameter_supported"].sum()
)

structural_supported = int(
    data["structural_supported"].sum()
)

print("\nPeriod-level support:")
print(data.to_string(index=False))

print("\n" + "=" * 80)
print("SUPPORT COUNTS")
print("=" * 80)

print(
    f"\nParameter-supported periods = "
    f"{parameter_supported}/{n_periods}"
)

print(
    f"Structural-supported periods = "
    f"{structural_supported}/{n_periods}"
)

# ------------------------------------------------------------
# Decision function
# ------------------------------------------------------------

def decision_for_requirement(required_periods):

    if structural_supported >= required_periods:
        return "REVISE"

    elif parameter_supported >= required_periods:
        return "RECALIBRATE"

    else:
        return "KEEP"


# ------------------------------------------------------------
# Test 1/3, 2/3, 3/3
# ------------------------------------------------------------

rows = []

for required in [1, 2, 3]:

    decision = decision_for_requirement(required)

    rows.append({
        "required_supported_periods": required,
        "required_fraction": f"{required}/{n_periods}",
        "parameter_support": parameter_supported,
        "structural_support": structural_supported,
        "decision": decision
    })

robustness = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("TEMPORAL REQUIREMENT SENSITIVITY")
print("=" * 80)

print(
    robustness.to_string(index=False)
)

# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

print("""
The purpose of this analysis is not to select a threshold
after observing the result. It evaluates how the real-data
decision changes under pre-specified temporal reproducibility
requirements.

A 1/3 rule asks whether a change is supported in at least one
period.

A 2/3 rule requires reproducibility across a majority of the
evaluated periods.

A 3/3 rule requires support in every evaluated period.

The current ETMR policy uses the 2/3 requirement.
""")

for _, row in robustness.iterrows():

    print(
        f"Requirement {row['required_fraction']}: "
        f"{row['decision']}"
    )

print("\nCurrent frozen policy = 2/3 supported periods.")

current = robustness[
    robustness["required_supported_periods"] == 2
]["decision"].iloc[0]

print(
    f"Current-policy real-data decision = {current}"
)

# ------------------------------------------------------------
# Stability assessment
# ------------------------------------------------------------

decisions = robustness["decision"].tolist()

if decisions[1] == decisions[2]:
    print(
        "\nPASS: The decision is stable between the "
        "current 2/3 requirement and the stricter 3/3 requirement."
    )
else:
    print(
        "\nNOTE: The decision changes between 2/3 and 3/3."
    )

if decisions[0] != decisions[1]:
    print(
        "IMPORTANT: Requiring reproducibility changes the "
        "decision relative to a one-period rule."
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

robustness.to_csv(
    "step87_temporal_robustness_results.csv",
    index=False
)

print("\nSaved:")
print("step87_temporal_robustness_results.csv")

print("\n" + "=" * 80)
print("STEP 87 COMPLETE")
print("=" * 80)
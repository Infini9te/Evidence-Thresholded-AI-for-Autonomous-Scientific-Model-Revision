import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 44: EVIDENCE SUFFICIENCY STRESS TEST")
print("=" * 80)

# =========================================================
# STRUCTURAL SIGNAL
# =========================================================

TEMPORAL = 0.414553
CONTEXT = 0.910836
LAGGED = 0.0
INTERACTION = 0.0


# =========================================================
# QUALITY LEVELS
# =========================================================

quality_levels = np.round(
    np.arange(1.00, 0.00, -0.05),
    2
)


# =========================================================
# FROZEN DIAGNOSTIC SCORES
# =========================================================

results = []

for q in quality_levels:

    # Evidence-adjusted structural support
    structural_support = (
        max(
            CONTEXT,
            LAGGED,
            INTERACTION
        )
        * q
    )

    # Evidence-adjusted persistence
    persistence_support = (
        TEMPORAL * q
    )

    # -----------------------------------------------------
    # Current ETMR decision
    # -----------------------------------------------------

    if q < 0.10:

        decision = "ABSTAIN"

    elif (
        CONTEXT >= 0.70
        and
        TEMPORAL >= 0.20
    ):

        decision = "REVISE"

    elif LAGGED >= 0.70:

        decision = "REVISE"

    elif INTERACTION >= 0.60:

        decision = "REVISE"

    elif TEMPORAL >= 0.70:

        decision = "RECALIBRATE"

    elif (
        TEMPORAL < 0.20
        and
        CONTEXT < 0.20
    ):

        decision = "KEEP"

    else:

        decision = "ABSTAIN"


    results.append({
        "Evidence_Quality": q,
        "Structural_Support": structural_support,
        "Persistence_Support": persistence_support,
        "Decision": decision
    })


df = pd.DataFrame(results)


# =========================================================
# DISPLAY
# =========================================================

print("\nStructural signal held constant:")
print(f"Temporal : {TEMPORAL}")
print(f"Context  : {CONTEXT}")

print("\n" + "=" * 80)
print("QUALITY STRESS TEST")
print("=" * 80)

print(
    df.to_string(index=False)
)


# =========================================================
# TRANSITION POINT
# =========================================================

revise_cases = df[
    df["Decision"] == "REVISE"
]

abstain_cases = df[
    df["Decision"] == "ABSTAIN"
]

print("\n" + "=" * 80)
print("DECISION TRANSITION")
print("=" * 80)

if len(revise_cases) > 0:
    print(
        "Lowest quality still producing REVISE:",
        revise_cases["Evidence_Quality"].min()
    )

if len(abstain_cases) > 0:
    print(
        "Highest quality producing ABSTAIN:",
        abstain_cases["Evidence_Quality"].max()
    )


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    "step44_evidence_sufficiency_stress.csv",
    index=False
)

print("\nSaved:")
print("step44_evidence_sufficiency_stress.csv")

print("\n" + "=" * 80)
print("STEP 44 COMPLETE")
print("=" * 80)
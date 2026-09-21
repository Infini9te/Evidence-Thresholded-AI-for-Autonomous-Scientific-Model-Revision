import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 11: DECISION ENGINE")
print("=" * 80)


# ============================================================
# LOAD WORLD COMPARISON
# ============================================================

df = pd.read_csv(
    "step10_world_comparison.csv"
)


# ============================================================
# DECISION FUNCTION
# ============================================================

def etmr_decision(row):

    mae = row["MAE"]
    temporal = row["Temporal"]
    context = row["Context"]
    provenance = row["Provenance"]

    # --------------------------------------------------------
    # STEP 1 — EVIDENCE QUALITY
    # --------------------------------------------------------

    if provenance < 0.10:

        return "ABSTAIN"

    # --------------------------------------------------------
    # STEP 2 — IS DISCREPANCY LARGE ENOUGH?
    # --------------------------------------------------------

    if mae < 0.0015:

        return "KEEP"

    # --------------------------------------------------------
    # STEP 3 — PARAMETER-LIKE ERROR
    # --------------------------------------------------------
    # Strong temporal persistence with weak/noisy
    # structural context is treated as calibration-like.

    if temporal > 0.70 and context >= 0.90:

        return "RECALIBRATE"

    # --------------------------------------------------------
    # STEP 4 — STRUCTURAL DISCREPANCY
    # --------------------------------------------------------

    if context > 0.70 and temporal > 0.20:

        return "REVISE"

    # --------------------------------------------------------
    # STEP 5 — OTHERWISE
    # --------------------------------------------------------

    return "ABSTAIN"


# ============================================================
# APPLY ETMR
# ============================================================

df["ETMR_Decision"] = df.apply(
    etmr_decision,
    axis=1
)


# ============================================================
# COMPARE WITH GROUND TRUTH
# ============================================================

df["Correct"] = (
    df["ETMR_Decision"]
    ==
    df["Expected_Decision"]
)


# ============================================================
# DISPLAY
# ============================================================

print()

print(
    df[
        [
            "World",
            "MAE",
            "Temporal",
            "Context",
            "Provenance",
            "Expected_Decision",
            "ETMR_Decision",
            "Correct"
        ]
    ].to_string(index=False)
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = df["Correct"].mean()

print()
print("-" * 80)

print(
    f"Decision accuracy : {accuracy:.2%}"
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "step11_etmr_decisions.csv",
    index=False
)


print()
print("=" * 80)
print("Saved: step11_etmr_decisions.csv")
print("=" * 80)
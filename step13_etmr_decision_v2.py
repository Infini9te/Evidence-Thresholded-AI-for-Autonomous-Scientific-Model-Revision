import pandas as pd

print("=" * 80)
print("ETMR — STEP 13: DECISION ENGINE v2")
print("=" * 80)


# ============================================================
# LOAD CONTROLLED BENCHMARK
# ============================================================

df = pd.read_csv(
    "step12_provenance_calibrated.csv"
)


# ============================================================
# ETMR DECISION FUNCTION
# ============================================================

def etmr_decision(row):

    mae = row["MAE"]
    temporal = row["Temporal"]
    context = row["Context"]
    provenance = row["Controlled_Provenance"]

    # --------------------------------------------------------
    # 1. EVIDENCE QUALITY GATE
    # --------------------------------------------------------
    #
    # If evidence quality is too low, ETMR must not make
    # a structural decision.
    #

    if provenance < 0.50:
        return "ABSTAIN"

    # --------------------------------------------------------
    # 2. SMALL DISCREPANCY
    # --------------------------------------------------------
    #
    # Small and non-persistent error → KEEP.
    #

    if mae < 0.0015 and temporal < 0.20:
        return "KEEP"

    # --------------------------------------------------------
    # 3. PARAMETER / CALIBRATION ERROR
    # --------------------------------------------------------
    #
    # Strong temporal persistence indicates a systematic
    # mismatch that may be explained by parameter values.
    #

    if temporal > 0.70:
        return "RECALIBRATE"

    # --------------------------------------------------------
    # 4. STRUCTURAL ERROR
    # --------------------------------------------------------
    #
    # Strong context dependence + temporal persistence
    # suggests that the current model structure is missing
    # an explanatory mechanism.
    #

    if context > 0.70 and temporal > 0.20:
        return "REVISE"

    # --------------------------------------------------------
    # 5. UNCERTAIN CASE
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
# CORRECTNESS
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
            "Controlled_Provenance",
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
    "step13_etmr_decisions_v2.csv",
    index=False
)


print()
print("=" * 80)
print("Saved: step13_etmr_decisions_v2.csv")
print("=" * 80)
import pandas as pd

print("=" * 80)
print("ETMR — STEP 12: PROVENANCE CALIBRATION")
print("=" * 80)


# ============================================================
# LOAD WORLD COMPARISON
# ============================================================

df = pd.read_csv(
    "step10_world_comparison.csv"
)


# ============================================================
# CONTROLLED PROVENANCE
# ============================================================
#
# Worlds A, B and C:
# trustworthy evidence
#
# World D:
# intentionally degraded evidence
# ============================================================

df["Controlled_Provenance"] = 1.0

df.loc[
    df["World"] == "D_Insufficient",
    "Controlled_Provenance"
] = 0.10


# ============================================================
# DISPLAY
# ============================================================

print()

print(
    df[
        [
            "World",
            "Provenance",
            "Controlled_Provenance",
            "Expected_Decision"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "step12_provenance_calibrated.csv",
    index=False
)


print()
print("=" * 80)
print("Saved: step12_provenance_calibrated.csv")
print("=" * 80)
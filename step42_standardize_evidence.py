import pandas as pd

print("=" * 80)
print("ETMR — STEP 42: STANDARDIZED EVIDENCE QUALITY")
print("=" * 80)

# =========================================================
# LOAD
# =========================================================

development = pd.read_csv(
    "step14_parameterized_benchmark.csv"
)

unseen = pd.read_csv(
    "step19_unseen_benchmark.csv"
)

rich = pd.read_csv(
    "step39_rich_etmr_results.csv"
)

# =========================================================
# STANDARDIZE QUALITY
# =========================================================

def standardize_basic(df):

    df = df.copy()

    # Controlled development cases:
    # A/B/C are high-quality evidence.
    # D already contains its designed evidence quality.
    if "World" in df.columns:

        def quality(row):

            world = str(row["World"])

            if world.startswith("D_"):
                return float(row["Provenance"])

            return 1.0

        df["Q_evidence"] = df.apply(
            quality,
            axis=1
        )

    else:
        df["Q_evidence"] = 1.0

    return df


development_std = standardize_basic(
    development
)

unseen_std = standardize_basic(
    unseen
)

rich_std = rich.copy()

# Rich unseen mechanisms are deliberately high-quality
rich_std["Q_evidence"] = 1.0


# =========================================================
# ADD RICH FEATURES WHERE ABSENT
# =========================================================

for df in [
    development_std,
    unseen_std,
    rich_std
]:

    if "Lagged_Rainfall" not in df.columns:
        df["Lagged_Rainfall"] = 0.0

    if "Interaction" not in df.columns:
        df["Interaction"] = 0.0


# =========================================================
# COMBINE
# =========================================================

combined = pd.concat(
    [
        development_std,
        unseen_std,
        rich_std
    ],
    ignore_index=True
)


# =========================================================
# DISPLAY
# =========================================================

print("\nSTANDARDIZED EVIDENCE QUALITY")
print("-" * 80)

print(
    combined[
        [
            "World",
            "Expected",
            "Temporal",
            "Context",
            "Lagged_Rainfall",
            "Interaction",
            "Provenance",
            "Q_evidence"
        ]
    ].to_string(index=False)
)


# =========================================================
# QUALITY BY CLASS
# =========================================================

print("\n" + "=" * 80)
print("QUALITY DISTRIBUTION BY DECISION")
print("=" * 80)

print(
    combined
    .groupby("Expected")["Q_evidence"]
    .agg(["count", "min", "max", "mean"])
)


# =========================================================
# SAVE
# =========================================================

combined.to_csv(
    "step42_standardized_evidence.csv",
    index=False
)

print("\nSaved:")
print("step42_standardized_evidence.csv")

print("\n" + "=" * 80)
print("STEP 42 COMPLETE")
print("=" * 80)
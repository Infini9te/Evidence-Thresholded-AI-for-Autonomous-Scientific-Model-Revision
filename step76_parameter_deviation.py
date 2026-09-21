import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 76: PARAMETER DEVIATION FROM KEEP BASELINE")
print("=" * 80)

df = pd.read_csv("step72_competing_evidence_results.csv")

# ---------------------------------------------------------
# KEEP defines the expected parameter region
# ---------------------------------------------------------

keep_alpha = df[df["truth"] == "KEEP"]["alpha_case"].values

keep_mean = keep_alpha.mean()
keep_std = keep_alpha.std(ddof=1)

keep_min = keep_alpha.min()
keep_max = keep_alpha.max()

q05 = np.quantile(keep_alpha, 0.05)
q95 = np.quantile(keep_alpha, 0.95)

print("\n" + "=" * 80)
print("KEEP PARAMETER BASELINE")
print("=" * 80)

print(f"KEEP alpha mean = {keep_mean:.6f}")
print(f"KEEP alpha std  = {keep_std:.6f}")
print(f"KEEP alpha min  = {keep_min:.6f}")
print(f"KEEP alpha max  = {keep_max:.6f}")
print(f"KEEP 5th pct    = {q05:.6f}")
print(f"KEEP 95th pct   = {q95:.6f}")

# ---------------------------------------------------------
# Absolute deviation from KEEP parameter expectation
# ---------------------------------------------------------

df["alpha_deviation"] = abs(
    df["alpha_case"] - keep_mean
)

# Relative deviation
df["alpha_relative_deviation"] = (
    abs(df["alpha_case"] - keep_mean) / abs(keep_mean)
)

# Standardized deviation
df["alpha_z"] = (
    df["alpha_case"] - keep_mean
) / keep_std

# ---------------------------------------------------------
# Print cases
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CASE-LEVEL PARAMETER DEVIATION")
print("=" * 80)

print(
    df[
        [
            "case",
            "truth",
            "alpha_case",
            "alpha_deviation",
            "alpha_relative_deviation",
            "alpha_z"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# Class summaries
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CLASS SUMMARY")
print("=" * 80)

summary = df.groupby("truth").agg(
    alpha_mean=("alpha_case", "mean"),
    alpha_min=("alpha_case", "min"),
    alpha_max=("alpha_case", "max"),
    deviation_mean=("alpha_deviation", "mean"),
    deviation_min=("alpha_deviation", "min"),
    deviation_max=("alpha_deviation", "max"),
    relative_deviation_mean=("alpha_relative_deviation", "mean"),
    relative_deviation_min=("alpha_relative_deviation", "min"),
    relative_deviation_max=("alpha_relative_deviation", "max")
)

print(summary.to_string())

# ---------------------------------------------------------
# Separation from KEEP parameter range
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("PARAMETER RANGE SEPARATION")
print("=" * 80)

for cls in ["RECALIBRATE", "REVISE"]:

    vals = df[df["truth"] == cls]["alpha_case"]

    below_keep = (vals < keep_min).sum()
    above_keep = (vals > keep_max).sum()
    inside_keep = ((vals >= keep_min) & (vals <= keep_max)).sum()

    print(
        f"{cls:15s}: "
        f"below KEEP range={below_keep}/{len(vals)}, "
        f"inside={inside_keep}/{len(vals)}, "
        f"above={above_keep}/{len(vals)}"
    )

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    "step76_parameter_deviation_results.csv",
    index=False
)

print("\nSaved:")
print("step76_parameter_deviation_results.csv")

print("\n" + "=" * 80)
print("STEP 76 COMPLETE")
print("=" * 80)
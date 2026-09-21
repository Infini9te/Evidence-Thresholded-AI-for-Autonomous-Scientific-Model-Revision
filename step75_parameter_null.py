import pandas as pd
import numpy as np

print("=" * 80)
print("ETMR — STEP 75: PARAMETER NULL DISTRIBUTION")
print("=" * 80)

df = pd.read_csv("step72_competing_evidence_results.csv")

# ---------------------------------------------------------
# KEEP cases define the empirical null distribution
# ---------------------------------------------------------

keep = df[df["truth"] == "KEEP"]["parameter_relative_gain"].values

null_mean = keep.mean()
null_std = keep.std(ddof=1)

null_q50 = np.quantile(keep, 0.50)
null_q90 = np.quantile(keep, 0.90)
null_q95 = np.quantile(keep, 0.95)
null_q99 = np.quantile(keep, 0.99)

print("\n" + "=" * 80)
print("KEEP PARAMETER NULL")
print("=" * 80)

print(f"Mean       = {null_mean:.6f}")
print(f"Std        = {null_std:.6f}")
print(f"Median     = {null_q50:.6f}")
print(f"90th pct   = {null_q90:.6f}")
print(f"95th pct   = {null_q95:.6f}")
print(f"99th pct   = {null_q99:.6f}")

# ---------------------------------------------------------
# Standardized parameter evidence
#
# Positive values:
#   improvement above the KEEP null
#
# Near zero:
#   compatible with ordinary KEEP behavior
# ---------------------------------------------------------

df["parameter_z"] = (
    df["parameter_relative_gain"] - null_mean
) / null_std

# Empirical exceedance relative to KEEP null
df["parameter_excess"] = (
    df["parameter_relative_gain"] - null_q95
)

# ---------------------------------------------------------
# Print cases
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CASE-LEVEL PARAMETER EVIDENCE")
print("=" * 80)

print(
    df[
        [
            "case",
            "truth",
            "parameter_relative_gain",
            "parameter_z",
            "parameter_excess"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# Class summary
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CLASS SUMMARY")
print("=" * 80)

summary = df.groupby("truth").agg(
    parameter_gain_mean=("parameter_relative_gain", "mean"),
    parameter_gain_min=("parameter_relative_gain", "min"),
    parameter_gain_max=("parameter_relative_gain", "max"),
    parameter_z_mean=("parameter_z", "mean"),
    parameter_z_min=("parameter_z", "min"),
    parameter_z_max=("parameter_z", "max")
)

print(summary.to_string())

# ---------------------------------------------------------
# Null exceedance checks
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("NULL EXCEEDANCE")
print("=" * 80)

for cls in ["KEEP", "RECALIBRATE", "REVISE"]:

    sub = df[df["truth"] == cls]

    above95 = (
        sub["parameter_relative_gain"] > null_q95
    ).sum()

    above99 = (
        sub["parameter_relative_gain"] > null_q99
    ).sum()

    print(
        f"{cls:15s}: "
        f">95th={above95}/{len(sub)}, "
        f">99th={above99}/{len(sub)}"
    )

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    "step75_parameter_null_results.csv",
    index=False
)

print("\nSaved:")
print("step75_parameter_null_results.csv")

print("\n" + "=" * 80)
print("STEP 75 COMPLETE")
print("=" * 80)
import pandas as pd
import numpy as np

print("=" * 70)
print("ETMR — CONTROLLED WORLD A: OBSERVATION NOISE")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD ORIGINAL DATA
# ------------------------------------------------------------

df = pd.read_csv("final_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ------------------------------------------------------------
# 2. ORIGINAL TARGET
# ------------------------------------------------------------

true_sm = df["sm_surface"].copy()

# ------------------------------------------------------------
# 3. ADD OBSERVATION NOISE
# ------------------------------------------------------------

np.random.seed(2026)

noise_std = true_sm.std() * 0.05

noise = np.random.normal(
    loc=0,
    scale=noise_std,
    size=len(df)
)

df["sm_surface_noisy"] = true_sm + noise

# ------------------------------------------------------------
# 4. RECORD THE GROUND TRUTH
# ------------------------------------------------------------

df["ground_truth_decision"] = "KEEP"

df["failure_type"] = "observation_noise"

df["noise_std"] = noise_std

# ------------------------------------------------------------
# 5. CHECK
# ------------------------------------------------------------

print()
print("Original SM mean :", true_sm.mean())
print("Original SM std  :", true_sm.std())

print()
print("Noise std        :", noise_std)

print()
print("Noisy SM mean    :", df["sm_surface_noisy"].mean())
print("Noisy SM std     :", df["sm_surface_noisy"].std())

print()
print("Ground-truth decision:")
print("KEEP")

# ------------------------------------------------------------
# 6. SAVE
# ------------------------------------------------------------

df.to_csv(
    "step5_world_A_noise.csv",
    index=False
)

print()
print("=" * 70)
print("Saved: step5_world_A_noise.csv")
print("=" * 70)
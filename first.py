import pandas as pd

df = pd.read_csv("final_dataset.csv")

print("Shape:", df.shape)
print("\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(i, col)

print("\nMissing values:")
print(df.isna().sum())

print("\nFill method:")
print(df["fill_method"].value_counts())

print("\nObservation flags:")
print(
    df[["is_observed", "is_interpolated"]]
    .value_counts()
)

print("\nDate range:")
print(df["date"].min(), "→", df["date"].max())
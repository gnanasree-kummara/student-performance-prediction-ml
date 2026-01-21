import pandas as pd

df = pd.read_csv("data/StudentPerformanceFactors.csv")


print("\n📌 Dataset Shape:", df.shape)
print("\n📌 First 5 Rows:\n", df.head())
print("\n📌 Column Names:\n", df.columns.tolist())
print("\n📌 Missing Values:\n", df.isna().sum())
print("\n📌 Data Types:\n", df.dtypes)

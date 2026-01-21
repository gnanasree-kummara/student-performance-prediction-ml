import pandas as pd

# ---- Load Dataset ----
df = pd.read_csv("data/StudentPerformanceFactors.csv")

print("====================================")
print("          DATASET LOADED")
print("====================================\n")

# ---- Basic Information ----
print("Dataset Shape (rows, columns):", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# ---- Missing Value Check ----
print("\nMissing Values:")
print(df.isnull().sum())

# ---- Statistical Summary ----
print("\nStatistical Summary (Numerical Features):")
print(df.describe())

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generate a reproducible sample customer dataset
np.random.seed(42)
n = 200

df = pd.DataFrame({
    "customer_id": range(1001, 1001 + n),
    "age": np.random.randint(18, 70, n).astype(float),
    "income": np.random.normal(45000, 12000, n).round(),
    "city": np.random.choice(
        ["Chennai", "chennai ", "Madurai", "Coimbatore"], n
    ),
    "purchases": np.random.poisson(6, n).astype(float),
    "churn": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
})

# Inject missing values, outliers, an invalid value and duplicate rows
df.loc[np.random.choice(n, 15, replace=False), "age"] = np.nan
df.loc[np.random.choice(n, 10, replace=False), "income"] = np.nan
df.loc[[5, 50, 120], "income"] = [450000, 520000, -3000]
df = pd.concat([df, df.iloc[[10, 20, 30, 40, 50]]], ignore_index=True)

print("CUSTOMER DATA: CLEANING AND SAMPLING")
print("=" * 48)
print("Raw dataset shape:", df.shape)

# 2. Detect missing values
print("\n[1] MISSING VALUES")
print(df.isnull().sum())

# 3. Detect and remove duplicate rows
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates().copy()
print(f"\n[2] Duplicate rows removed: {duplicate_count}")
print("Shape after removing duplicates:", df.shape)

# 4. Standardise inconsistent category names
print("\n[3] City values before cleaning:")
print(sorted(df["city"].unique()))
df["city"] = df["city"].str.strip().str.title()
print("City values after cleaning:")
print(sorted(df["city"].unique()))

df.loc[df["income"] < 0, "income"] = np.nan

q1, q3 = df["income"].quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[
    (df["income"] < lower_bound) | (df["income"] > upper_bound)
]
print(f"\n[4] Income IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")
print("Number of income outliers:", len(outliers))
print(outliers[["customer_id", "income"]].to_string(index=False))

df["income"] = df["income"].clip(lower_bound, upper_bound)


df["age"] = df["age"].fillna(df["age"].median())
df["income"] = df["income"].fillna(df["income"].mean()).round()
print("\n[5] Missing values after imputation:")
print(df.isnull().sum())


df["churn"] = df["churn"].map({"No": 0, "Yes": 1})
df = pd.get_dummies(df, columns=["city"], drop_first=True, dtype=int)

# Standardise numeric input features
scaler = StandardScaler()
numeric_features = ["age", "income", "purchases"]
df[numeric_features] = scaler.fit_transform(df[numeric_features])

print("\n[6] CLEANED AND TRANSFORMED DATA (FIRST FIVE ROWS)")
print(df.head().round(3).to_string(index=False))

# 8. Draw a simple random sample of 20%
simple_sample = df.sample(frac=0.20, random_state=1)
print("\n[7] SIMPLE RANDOM SAMPLE")
print("Rows selected:", len(simple_sample))
print("Churn rate:", round(simple_sample["churn"].mean(), 3))

# 9. Stratified 70/15/15 train, validation and test split
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=1
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=1
)

print("\nSTRATIFIED TRAIN / VALIDATION / TEST SPLIT")
for name, target in [
    ("Overall", y),
    ("Train", y_train),
    ("Validation", y_validation),
    ("Test", y_test)
]:
    print(
        f"{name:12s} rows = {len(target):3d} "
        f"churn rate = {target.mean():.3f}"
    )


df.to_csv("cleaned_customer_data.csv", index=False)
X_train.assign(churn=y_train).to_csv("train_data.csv", index=False)
X_validation.assign(churn=y_validation).to_csv(
    "validation_data.csv", index=False
)
X_test.assign(churn=y_test).to_csv("test_data.csv", index=False)

print("\nExperiment 4 completed. Cleaned data and split files saved.")

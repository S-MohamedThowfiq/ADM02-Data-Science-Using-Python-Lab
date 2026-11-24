
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generate a reproducible sample student dataset
np.random.seed(7)
n = 60

students = pd.DataFrame({
    "roll_no": range(1, n + 1),
    "gender": np.random.choice(["Male", "Female"], n),
    "department": np.random.choice(
        ["CSE", "ECE", "MECH", "AI&DS"], n,
        p=[0.30, 0.25, 0.20, 0.25]
    ),
    "grade": np.random.choice(
        ["A", "B", "C", "D"], n,
        p=[0.25, 0.40, 0.25, 0.10]
    ),
    "attendance": np.random.randint(60, 100, n),
    "study_hours": np.round(np.random.uniform(1, 8, n), 1)
})

students["marks"] = np.clip(
    (
        35 + 6 * students["study_hours"]
        + 0.2 * students["attendance"]
        + np.random.normal(0, 6, n)
    ).round(),
    0, 100
).astype(int)

print("STUDENT PERFORMANCE: EXPLORATORY DATA ANALYSIS")
print("=" * 52)
print("\nFirst five records:")
print(students.head().to_string(index=False))

# 2. Classify the attributes
attribute_types = {
    "roll_no": "Nominal (identifier)",
    "gender": "Qualitative - Nominal",
    "department": "Qualitative - Nominal",
    "grade": "Qualitative - Ordinal",
    "attendance": "Quantitative - Discrete",
    "study_hours": "Quantitative - Continuous",
    "marks": "Quantitative - Discrete"
}

print("\nATTRIBUTE CLASSIFICATION")
for column, kind in attribute_types.items():
    print(f"{column:15s}: {kind}")

# 3. Descriptive statistics
numeric = students[["attendance", "study_hours", "marks"]]

print("\nDESCRIPTIVE STATISTICS")
print(numeric.describe().round(2).to_string())
print("\nMean:\n", numeric.mean().round(2))
print("\nMedian:\n", numeric.median().round(2))
print("\nMode of each numeric attribute:\n", numeric.mode().iloc[0])
print("\nStandard deviation:\n", numeric.std().round(2))
print("\nQuartiles:\n", numeric.quantile([0.25, 0.50, 0.75]).round(2))
print("\nSkewness:\n", numeric.skew().round(3))

# 4. Frequency table, cross-tabulation and correlation
print("\nFREQUENCY TABLE - DEPARTMENT")
print(students["department"].value_counts())

print("\nCROSS-TABULATION - DEPARTMENT VS GRADE")
print(pd.crosstab(students["department"], students["grade"]))

print("\nCORRELATION MATRIX")
print(numeric.corr().round(3))

# 5. Draw histogram, bar chart, pie chart and scatter plot
fig, ax = plt.subplots(2, 2, figsize=(11, 8))

ax[0, 0].hist(students["marks"], bins=8, edgecolor="black")
ax[0, 0].set_title("Histogram of Marks")
ax[0, 0].set_xlabel("Marks")
ax[0, 0].set_ylabel("Frequency")

department_counts = students["department"].value_counts()
ax[0, 1].bar(department_counts.index, department_counts.values)
ax[0, 1].set_title("Students per Department")
ax[0, 1].set_xlabel("Department")
ax[0, 1].set_ylabel("Number of Students")

grade_counts = students["grade"].value_counts().reindex(["A", "B", "C", "D"])
ax[1, 0].pie(
    grade_counts.values, labels=grade_counts.index,
    autopct="%1.1f%%", startangle=90
)
ax[1, 0].set_title("Grade Distribution")

ax[1, 1].scatter(students["study_hours"], students["marks"])
ax[1, 1].set_title("Study Hours vs Marks")
ax[1, 1].set_xlabel("Study hours per day")
ax[1, 1].set_ylabel("Marks")

plt.tight_layout()
plt.savefig("exp03_eda.png", dpi=150)
plt.show()

# 6. Interpret the relationship between study hours and marks
correlation = students["study_hours"].corr(students["marks"])
print(f"\nCorrelation between study hours and marks: {correlation:.3f}")
if correlation > 0:
    print("Interpretation: Study hours and marks have a positive relationship.")
elif correlation < 0:
    print("Interpretation: Study hours and marks have a negative relationship.")
else:
    print("Interpretation: No linear relationship is observed.")

print("\nGraph saved as exp03_eda.png")

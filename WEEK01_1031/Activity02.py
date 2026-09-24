# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Activity 2: Student Performance Tracker

# 1. Generate random data for 100 students
np.random.seed(42)

data = {
    "ID": range(1, 101),
    "Math": np.random.randint(40, 101, 100),
    "Science": np.random.randint(40, 101, 100),
    "English": np.random.randint(40, 101, 100),
    "Grade": np.random.choice(["A", "B", "C"], 100)
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df.head(10))

# 2. Add some missing marks for demonstration
df.loc[5, "Math"] = np.nan
df.loc[15, "Science"] = np.nan
df.loc[25, "English"] = np.nan

print("\nDataset with missing marks:")
print(df.loc[[5, 15, 25]])

# 3. Handle missing marks by filling with the mean
subjects = ["Math", "Science", "English"]

for subject in subjects:
    mean_value = df[subject].mean()
    df[subject] = df[subject].fillna(mean_value)

print("\nDataset after filling missing marks with mean:")
print(df.loc[[5, 15, 25]])

# 4. Encode the Grade column
label_encoder = LabelEncoder()

df["Grade_Encoded"] = label_encoder.fit_transform(df["Grade"])

print("\nAfter Label Encoding:")
print(df.head(10))

print("\nGrade Mapping:")
for number, grade in enumerate(label_encoder.classes_):
    print(number, "=", grade)

# 5. Calculate total score
df["Total"] = df["Math"] + df["Science"] + df["English"]

# 6. Histogram of Math scores with 10 bins
plt.figure(figsize=(8, 5))

plt.hist(df["Math"], bins=10, edgecolor="black")

plt.xlabel("Math Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Math Scores")
plt.show()

# 7. Boxplot for total scores
plt.figure(figsize=(8, 5))

plt.boxplot(df["Total"])

plt.ylabel("Total Score")
plt.title("Boxplot of Student Total Scores")
plt.show()

# Find unusually high/low total scores using IQR
Q1 = df["Total"].quantile(0.25)
Q3 = df["Total"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Total"] < lower_bound) |
    (df["Total"] > upper_bound)
]

print("\nTotal Score Outlier Detection:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

if outliers.empty:
    print("No students have unusually high or low total scores.")
else:
    print("\nStudents with unusually high/low total scores:")
    print(outliers[["ID", "Math", "Science", "English", "Grade", "Total"]])

# 8. Summarize findings
print("\n--- Summary of Findings ---")

# Average total score for each grade
grade_performance = df.groupby("Grade")["Total"].mean().sort_values(ascending=False)

print("\nAverage Total Score by Grade:")
print(grade_performance)

best_grade = grade_performance.idxmax()

print("\nGrade that performed best overall:", best_grade)

print("\nAverage subject scores:")
print("Math:", round(df["Math"].mean(), 2))
print("Science:", round(df["Science"].mean(), 2))
print("English:", round(df["English"].mean(), 2))

print("\nNumber of students in each grade:")
print(df["Grade"].value_counts())
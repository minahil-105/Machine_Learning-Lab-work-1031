# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import pandas as pd
import matplotlib.pyplot as plt

# Activity 4: Multi-format Data Loading & Visualization

# 1. Load the three files

students = pd.read_csv("students.csv")
attendance = pd.read_json("attendance.json")
extra = pd.read_excel("extra.xlsx")

print("Students Data:")
print(students)

print("\nAttendance Data:")
print(attendance)

print("\nBonus Marks Data:")
print(extra)


# 2. Rename columns if necessary
# Make sure all files use the same student name column

students.columns = ["Name", "Marks"]
attendance.columns = ["Name", "Attendance"]
extra.columns = ["Name", "Bonus"]


# 3. Merge all three DataFrames

df = pd.merge(students, attendance, on="Name", how="inner")
df = pd.merge(df, extra, on="Name", how="inner")

print("\nMerged DataFrame:")
print(df)


# 4. Create scatter plot

# Students with attendance below 70%
low_attendance = df[df["Attendance"] < 70]
normal_attendance = df[df["Attendance"] >= 70]

plt.figure(figsize=(8, 6))

# Normal attendance students
plt.scatter(
    normal_attendance["Marks"],
    normal_attendance["Attendance"],
    label="Attendance >= 70%"
)

# Low attendance students
plt.scatter(
    low_attendance["Marks"],
    low_attendance["Attendance"],
    label="Attendance < 70%"
)

# Add student names to the points
for i in range(len(df)):
    plt.annotate(
        df["Name"].iloc[i],
        (
            df["Marks"].iloc[i],
            df["Attendance"].iloc[i]
        )
    )

plt.xlabel("Marks")
plt.ylabel("Attendance (%)")
plt.title("Marks vs Attendance")
plt.legend()
plt.grid(True)
plt.show()


# 5. One-Hot Encoding on Name

df_encoded = pd.get_dummies(
    df,
    columns=["Name"],
    dtype=int
)

print("\nDataFrame after One-Hot Encoding:")
print(df_encoded)
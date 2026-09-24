import pandas as pd
import matplotlib.pyplot as plt

# Activity 3: COVID-19 Dataset Exploration

# 1. Load the online dataset
url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"

df = pd.read_csv(url)

print("Dataset loaded successfully!")
print(df.head())

# 2. Extract only Pakistan's data
pakistan = df[df["Country/Region"] == "Pakistan"].copy()

print("\nPakistan COVID-19 Data:")
print(pakistan.head())

# 3. Handle missing values
print("\nMissing values before handling:")
print(pakistan.isnull().sum())

# Fill missing numeric values using forward fill,
# then backward fill if any values remain
numeric_columns = ["Confirmed", "Recovered", "Deaths"]

pakistan[numeric_columns] = pakistan[numeric_columns].ffill()
pakistan[numeric_columns] = pakistan[numeric_columns].bfill()

print("\nMissing values after handling:")
print(pakistan.isnull().sum())

# 4. Calculate daily new confirmed cases
pakistan["Daily Cases"] = pakistan["Confirmed"].diff()

# The first day has no previous day to compare with
pakistan["Daily Cases"] = pakistan["Daily Cases"].fillna(0)

# Avoid negative values caused by data corrections
pakistan["Daily Cases"] = pakistan["Daily Cases"].clip(lower=0)

# 5. Plot confirmed cases over time
plt.figure(figsize=(12, 6))

plt.plot(
    pakistan["Date"],
    pakistan["Confirmed"],
    label="Confirmed Cases"
)

plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.title("COVID-19 Confirmed Cases in Pakistan")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# 6. Find the day with the highest confirmed cases
highest_confirmed = pakistan.loc[
    pakistan["Confirmed"].idxmax()
]

print("\nDay with the highest confirmed cases:")
print("Date:", highest_confirmed["Date"])
print("Confirmed Cases:", highest_confirmed["Confirmed"])

# 7. Detect outliers in daily cases using a boxplot
plt.figure(figsize=(8, 6))

plt.boxplot(pakistan["Daily Cases"])

plt.ylabel("Daily Confirmed Cases")
plt.title("Boxplot of Daily COVID-19 Cases in Pakistan")
plt.grid(True)

plt.show()

# 8. Identify outliers using IQR
Q1 = pakistan["Daily Cases"].quantile(0.25)
Q3 = pakistan["Daily Cases"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = pakistan[
    (pakistan["Daily Cases"] < lower_bound) |
    (pakistan["Daily Cases"] > upper_bound)
]

print("\nOutlier Detection:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

if outliers.empty:
    print("No outliers were detected in daily cases.")
else:
    print("\nOutlier days:")
    print(
        outliers[
            ["Date", "Confirmed", "Daily Cases"]
        ].to_string(index=False)
    )

print("\nAnalysis completed successfully!")
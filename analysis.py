# analysis.py
"""
Titanic Dataset Analysis with SQL-style Queries
Author: Your Name
Description: Data exploration, visualization, and SQL-style queries on Titanic dataset.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset directly from seaborn
df = sns.load_dataset("titanic")

# ----- Data Overview -----
print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# ----- Exploratory Analysis -----
# Overall survival rate
survival_rate = df["survived"].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.2f}%")

# Survival rate by gender
gender_survival = df.groupby("sex")["survived"].mean() * 100
print("\nSurvival Rate by Gender:\n", gender_survival)

# Survival rate by class
class_survival = df.groupby("class")["survived"].mean() * 100
print("\nSurvival Rate by Class:\n", class_survival)

# ----- SQL-Style Analysis with pandas -----
print("\n--- SQL-Style Queries using pandas ---")

# 1. Equivalent to:
# SELECT sex, COUNT(*) AS count FROM passengers GROUP BY sex;
sex_count = df.groupby("sex").size().reset_index(name="count")
print("\nPassenger count by gender:\n", sex_count)

# 2. Equivalent to:
# SELECT class, AVG(age) FROM passengers GROUP BY class;
avg_age_class = df.groupby("class")["age"].mean().reset_index()
print("\nAverage Age by Class:\n", avg_age_class)

# 3. Equivalent to:
# SELECT * FROM passengers WHERE age < 18 AND survived = 1;
children_survivors = df.query("age < 18 and survived == 1")
print("\nChildren Survivors (Age < 18):\n", children_survivors.head())

# 4. Equivalent to:
# SELECT class, sex, AVG(survived) as survival_rate
# FROM passengers GROUP BY class, sex;
survival_by_class_gender = df.groupby(["class", "sex"])["survived"].mean().reset_index()
print("\nSurvival Rate by Class & Gender:\n", survival_by_class_gender)

# ----- Visualizations -----
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Survival Count by Gender")
plt.show()

sns.barplot(x="class", y="survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

sns.histplot(df["age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()

# analysis.py
"""
Titanic Dataset Analysis
Author: Abhishek
Description: Basic data exploration and visualization on the Titanic dataset.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset directly from seaborn (no CSV needed)
df = sns.load_dataset("titanic")

# ----- Data Overview -----
print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# ----- Exploratory Analysis -----
# Survival rate
survival_rate = df["survived"].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.2f}%")

# Group by gender
gender_survival = df.groupby("sex")["survived"].mean() * 100
print("\nSurvival Rate by Gender:\n", gender_survival)

# Group by class
class_survival = df.groupby("class")["survived"].mean() * 100
print("\nSurvival Rate by Class:\n", class_survival)

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

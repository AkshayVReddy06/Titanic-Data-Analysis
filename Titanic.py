import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# -----------------------------
# DATA CLEANING
# -----------------------------

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with most frequent value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop('Cabin', axis=1, inplace=True)

# -----------------------------
# ANALYSIS
# -----------------------------

# 1. Survival by Gender
gender_survival = df.groupby('Sex')['Survived'].mean()
print("\nSurvival Rate by Gender:\n", gender_survival)

# 2. Survival by Passenger Class
class_survival = df.groupby('Pclass')['Survived'].mean()
print("\nSurvival Rate by Class:\n", class_survival)

# 3. Survival by Age Group
bins = [0, 12, 20, 40, 60, 100]
labels = ['Child', 'Teen', 'Adult', 'Mid-age', 'Senior']

df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

age_survival = df.groupby('AgeGroup')['Survived'].mean()
print("\nSurvival Rate by Age Group:\n", age_survival)

# -----------------------------
# VISUALIZATION
# -----------------------------

# Style
sns.set(style="whitegrid")

# 1. Survival by Gender
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

# 2. Survival by Class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

# 3. Age Distribution
sns.histplot(df['Age'], bins=30)
plt.title("Age Distribution")
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\khush\OneDrive\Desktop\tested.csv")
print(df)

#printing starting 5 rows of the dataset
print(df.head())

#information about the given dataset
print(df.info())

#checking for missing values
print(df.isnull().sum())

#describing the nuemerical columns
print(df.describe())

# data distributions using histogram
plt.figure(figsize=(7, 4))
plt.hist(df['Fare'], bins=30, color='skyblue', edgecolor='black')
plt.title('fare histogram')
plt.xlabel('Fare')
plt.ylabel('Npassengers')
plt.grid(True)
plt.show()

# boxplot to idenitfy the outliers

plt.figure(figsize=(4, 6))
sns.boxplot(y=df['Fare'], color='red')
plt.title('fare boxplot')
plt.show()


sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare vs Passenger Class")
plt.show()

#corelation using heatmap

df_corr = df[['Survived', 'Pclass', 'Fare', 'PassengerId']].corr()
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title("Full Correlation Heatmap")
plt.show()

#value counts for the categorical columns
print("Pclass Value Counts:")
print(df['Pclass'].value_counts())

print("\nEmbarked Value Counts:")
print(df['Embarked'].value_counts())

print("\nSurvived Value Counts:")
print(df['Survived'].value_counts())

# countplot for categorical features

sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()

sns.countplot(x='Embarked', data=df)
plt.title("Port of Embarkation Distribution")
plt.show()

sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# using z _score to detect outliers in data

from scipy.stats import zscore
df['Fare_z'] = zscore(df['Fare'])
outliers = df[np.abs(df['Fare_z']) > 3]
print("outliers in fare", len(outliers))




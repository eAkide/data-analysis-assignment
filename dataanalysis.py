# data_analysis_assignment.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset with error handling
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully.\n")
except Exception as e:
    print(f" Error loading dataset: {e}")
    exit()

# Display first few rows
print("first 5 rows of the dataset:")
print(df.head(), "\n")

# Check structure and data types
print("Dataset Info:")
print(df.info(), "\n")

# Check for missing values
print(" Missing Values Check:")
print(df.isnull().sum(), "\n")

# Clean data (fill or drop missing values — none expected in Iris)
# Example of dropping missing values if there were any:
df.dropna(inplace=True)

# =========================
# Basic Data Analysis
# =========================

print("Descriptive Statistics:")
print(df.describe(), "\n")

# Group by species and compute mean of numeric columns
print("Mean values per species:")
print(df.groupby('target').mean(), "\n")

# Map target numbers to species names
df['species'] = df['target'].map(dict(zip(range(3), iris.target_names)))

# =========================
# Data Visualization
# =========================

# Set style
sns.set(style="whitegrid")

# 1. Line chart: average sepal length per species
species_means = df.groupby('species')['sepal length (cm)'].mean()
species_means.plot(kind='line', marker='o')
plt.title("Average Sepal Length per Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar chart: average petal length per species
species_petal = df.groupby('species')['petal length (cm)'].mean()
species_petal.plot(kind='bar', color='orange')
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram: distribution of sepal width
plt.hist(df['sepal width (cm)'], bins=15, color='green', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter plot: sepal length vs petal length
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c=df['target'], cmap='viridis')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label='Species')
plt.tight_layout()
plt.show()

# =========================
# Observations
# =========================
print("\n🔍 Observations:")
print("- No missing data was found.")
print("- Setosa species has generally shorter petals compared to others.")
print("- Clear correlation between sepal length and petal length.")
print("- Petal length increases as species change from Setosa to Virginica.")
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of each feature
df.hist(figsize=(10, 8))
plt.suptitle('Histograms of Iris Features')
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df, hue='target')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

# Load the Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='PRICE')

# Combine features and target into a single DataFrame
df = pd.concat([X, y], axis=1)

# Basic statistical summary
print(df.describe())

# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Boston Housing Dataset')
plt.show()

# Distribution of target variable (PRICE)
plt.figure(figsize=(10, 6))
sns.histplot(df['PRICE'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.show()

# Pairplot of some important features
features_of_interest = ['RM', 'LSTAT', 'PTRATIO', 'PRICE']
sns.pairplot(df[features_of_interest])
plt.suptitle('Pairplot of Key Features', y=1.02)
plt.show()

# Boxplot of PRICE vs CHAS (Charles River dummy variable)
plt.figure(figsize=(8, 6))
sns.boxplot(x='CHAS', y='PRICE', data=df)
plt.title('House Prices vs Charles River Location')
plt.show()

# Scatter plot of PRICE vs LSTAT (% lower status of the population)
plt.figure(figsize=(10, 6))
plt.scatter(df['LSTAT'], df['PRICE'])
plt.title('House Prices vs % Lower Status of the Population')
plt.xlabel('% Lower Status of the Population')
plt.ylabel('Price')
plt.show()

# Scatter plot of PRICE vs RM (average number of rooms)
plt.figure(figsize=(10, 6))
plt.scatter(df['RM'], df['PRICE'])
plt.title('House Prices vs Average Number of Rooms')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Price')
plt.show()

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='PRICE')

# Feature Engineering

# 1. Create interaction terms
X['RM_LSTAT'] = X['RM'] * X['LSTAT']
X['LSTAT_PTRATIO'] = X['LSTAT'] * X['PTRATIO']

# 2. Create polynomial features for important variables
X['RM_sq'] = X['RM'] ** 2
X['LSTAT_sq'] = X['LSTAT'] ** 2

# 3. Log transform for skewed features
X['LOG_CRIM'] = np.log1p(X['CRIM'])
X['LOG_ZN'] = np.log1p(X['ZN'])
X['LOG_B'] = np.log1p(X['B'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline Model: Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Baseline Model Performance:")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(model.coef_)
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.show()

# Residual Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=y_test - y_pred)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# Print feature coefficients
coefficients = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
})
print("\nFeature Coefficients:")
print(coefficients.sort_values('coefficient', key=abs, ascending=False))

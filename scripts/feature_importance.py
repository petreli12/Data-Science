import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

# Load the Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='PRICE')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=boston.feature_names)
plt.title("Feature Importance (SHAP Values)")
plt.tight_layout()
plt.show()

# Detailed SHAP plot for top 10 features
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=boston.feature_names, max_display=10)
plt.title("SHAP Summary Plot (Top 10 Features)")
plt.tight_layout()
plt.show()

# SHAP dependence plot for the most important feature
most_important_feature = boston.feature_names[np.argmax(np.abs(shap_values).mean(0))]
plt.figure(figsize=(10, 6))
shap.dependence_plot(most_important_feature, shap_values, X_test, feature_names=boston.feature_names)
plt.title(f"SHAP Dependence Plot for {most_important_feature}")
plt.tight_layout()
plt.show()

# SHAP force plot for a single prediction
sample_idx = 0
plt.figure(figsize=(12, 4))
shap.force_plot(explainer.expected_value, shap_values[sample_idx], X_test.iloc[sample_idx], feature_names=boston.feature_names, matplotlib=True, show=False)
plt.title(f"SHAP Force Plot for Sample {sample_idx}")
plt.tight_layout()
plt.show()

# Print feature importance based on SHAP values
feature_importance = pd.DataFrame({
    'feature': boston.feature_names,
    'importance': np.abs(shap_values).mean(0)
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance based on SHAP values:")
print(feature_importance)

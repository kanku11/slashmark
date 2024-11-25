import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Simulate a dataset (use your own dataset here if available)
def generate_insurance_data(size=1000):
    np.random.seed(42)
    age = np.random.randint(18, 70, size)
    income = np.random.randint(20000, 150000, size)
    health_score = np.random.randint(1, 10, size)
    claim_history = np.random.randint(0, 5, size)
    vehicle_age = np.random.randint(0, 15, size)
    premium = (
        0.1 * income
        + 100 * (70 - age)
        + 500 * (10 - health_score)
        + 300 * claim_history
        - 50 * vehicle_age
        + np.random.normal(0, 200, size)  # Noise
    )
    data = pd.DataFrame({
        "Age": age,
        "Income": income,
        "HealthScore": health_score,
        "ClaimHistory": claim_history,
        "VehicleAge": vehicle_age,
        "Premium": premium,
    })
    return data

# Generate dataset
data = generate_insurance_data(1000)

# Data exploration
print(data.head())
sns.pairplot(data)
plt.show()

# Feature and target split
X = data[["Age", "Income", "HealthScore", "ClaimHistory", "VehicleAge"]]
y = data["Premium"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualize predictions vs actuals
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Premium")
plt.ylabel("Predicted Premium")
plt.title("Actual vs Predicted Premium")
plt.show()

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)
feature_importance.plot(kind="bar", color="skyblue")
plt.title("Feature Importance")
plt.show()

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("dataset/cleaned_NVDA.csv")

# Features and Target
X = df[['open', 'high', 'low', 'volume']]
y = df['close']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train Model
rf_model.fit(X_train, y_train)

# Prediction
y_pred = rf_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)

# Save model
joblib.dump(rf_model, "models/random_forest_model.pkl")

print("Model saved successfully!")


plt.figure(figsize=(12,6))
plt.plot(y_test.values[:100],label='Actual Price')
plt.plot(y_pred[:100],label='Predicted Price')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Samples')
plt.ylabel('Price')
plt.legend()
plt.show()
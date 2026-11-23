from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the built-in California housing dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target  # Prices are measured in units of $100,000

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train a Random Forest regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate predictions
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("House Price Prediction")
print("----------------------")
print(f"Mean Absolute Error: ${mae * 100_000:,.0f}")
print(f"Root Mean Squared Error: ${rmse * 100_000:,.0f}")
print(f"R² Score: {r2:.3f}")

# Predict a sample house using the first test example
sample_house = X_test[0].reshape(1, -1)
predicted_value = model.predict(sample_house)[0]

print("\nExample prediction:")
print(f"Predicted house value: ${predicted_value * 100_000:,.0f}")
print(f"Actual house value:    ${y_test[0] * 100_000:,.0f}")

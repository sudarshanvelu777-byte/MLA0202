
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([10, 20, 25, 30, 40, 45, 50, 60, 65, 75])

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create Linear Regression model
model = LinearRegression()

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict values
y_pred = model.predict(X_test)

# 6. Display model parameters
print("Linear Regression Results")
print("--------------------------")
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# 7. Display actual and predicted values
print("\nActual Values:", y_test)
print("Predicted Values:", np.round(y_pred, 2))

# 8. Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", round(mse, 2))
print("R-squared Score:", round(r2, 4))

# 9. Plot the dataset and regression line
plt.scatter(X, y, color="blue", label="Actual Data")

X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)

plt.plot(X_line, y_line, color="red", label="Regression Line")

plt.xlabel("X - Independent Variable")
plt.ylabel("Y - Dependent Variable")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# 10. Predict a new value
new_X = np.array([[11]])
prediction = model.predict(new_X)

print("\nPrediction for X = 11:", round(prediction[0], 2))

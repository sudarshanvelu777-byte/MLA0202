
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Create a sample nonlinear dataset
np.random.seed(42)
X = np.linspace(1, 10, 50).reshape(-1, 1)
y = 2 * X.ravel()**2 + 3 * X.ravel() + 5
y = y + np.random.normal(0, 15, size=len(y))

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# 4. Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_pred = poly_model.predict(X_test_poly)

# 5. Evaluate both models
print("LINEAR REGRESSION")
print("-----------------")
print("MSE:", round(mean_squared_error(y_test, linear_pred), 2))
print("R2 Score:", round(r2_score(y_test, linear_pred), 4))

print("\nPOLYNOMIAL REGRESSION")
print("---------------------")
print("MSE:", round(mean_squared_error(y_test, poly_pred), 2))
print("R2 Score:", round(r2_score(y_test, poly_pred), 4))

# 6. Generate smooth curves for plotting
X_plot = np.linspace(1, 10, 200).reshape(-1, 1)
linear_curve = linear_model.predict(X_plot)
poly_curve = poly_model.predict(poly.transform(X_plot))

# 7. Plot comparison
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Actual Data")
plt.plot(X_plot, linear_curve, label="Linear Regression")
plt.plot(X_plot, poly_curve, label="Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()

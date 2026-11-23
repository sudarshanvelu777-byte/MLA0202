import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Fictional sample data for demonstration
data = pd.DataFrame({
    "brand": [
        "Toyota", "Honda", "Ford", "Toyota", "BMW", "Honda",
        "Ford", "BMW", "Toyota", "Honda", "Ford", "BMW",
        "Toyota", "Honda", "Ford", "BMW", "Toyota", "Honda",
        "Ford", "BMW"
    ],
    "year": [
        2018, 2019, 2017, 2020, 2021, 2016, 2019, 2018, 2015, 2022,
        2016, 2020, 2021, 2017, 2022, 2019, 2014, 2020, 2018, 2022
    ],
    "mileage_km": [
        45000, 35000, 70000, 25000, 18000, 85000, 40000, 50000, 95000, 12000,
        80000, 30000, 22000, 65000, 15000, 38000, 120000, 28000, 55000, 10000
    ],
    "fuel": [
        "Petrol", "Petrol", "Diesel", "Hybrid", "Petrol", "Petrol",
        "Diesel", "Diesel", "Petrol", "Hybrid", "Diesel", "Petrol",
        "Hybrid", "Petrol", "Petrol", "Diesel", "Petrol", "Hybrid",
        "Diesel", "Petrol"
    ],
    "transmission": [
        "Automatic", "Manual", "Manual", "Automatic", "Automatic",
        "Manual", "Automatic", "Automatic", "Manual", "Automatic",
        "Manual", "Automatic", "Automatic", "Manual", "Automatic",
        "Automatic", "Manual", "Automatic", "Manual", "Automatic"
    ],
    "price": [
        18000, 20000, 14000, 26000, 48000, 11000, 22000, 35000, 9000, 28000,
        12000, 42000, 30000, 13000, 27000, 39000, 7000, 27000, 17000, 50000
    ],
})

target = "price"
X = data.drop(columns=[target])
y = data[target]

categorical_columns = ["brand", "fuel", "transmission"]
numeric_columns = ["year", "mileage_km"]

# Compatible with older and newer scikit-learn versions
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocessor = ColumnTransformer([
    ("numbers", SimpleImputer(strategy="median"), numeric_columns),
    ("categories", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ]), categorical_columns),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Car Price Prediction Model")
print("--------------------------")
print(f"Mean absolute error: ${mean_absolute_error(y_test, predictions):,.2f}")
print(f"R² score: {r2_score(y_test, predictions):.2f}")

# Predict the price of a new car
new_car = pd.DataFrame([{
    "brand": "Toyota",
    "year": 2020,
    "mileage_km": 30000,
    "fuel": "Hybrid",
    "transmission": "Automatic",
}])

predicted_price = model.predict(new_car)[0]
print(f"\nEstimated price for the example car: ${predicted_price:,.2f}")

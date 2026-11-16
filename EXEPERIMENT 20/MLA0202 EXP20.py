import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


# Example monthly sales data (fictional)
months = pd.date_range(start="2021-01-01", periods=48, freq="MS")
rng = np.random.default_rng(42)

sales = (
    1000
    + np.arange(len(months)) * 25          # upward trend
    + 180 * np.sin(2 * np.pi * np.arange(len(months)) / 12)  # seasonality
    + rng.normal(0, 80, len(months))        # random variation
)

data = pd.DataFrame({"date": months, "sales": sales})


# Create features using only information available before each prediction.
data["month"] = data["date"].dt.month
data["time_index"] = np.arange(len(data))
data["lag_1"] = data["sales"].shift(1)
data["lag_2"] = data["sales"].shift(2)
data["lag_3"] = data["sales"].shift(3)
data["rolling_mean_3"] = data["sales"].shift(1).rolling(window=3).mean()

model_data = data.dropna().copy()

feature_columns = [
    "month",
    "time_index",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
]

X = model_data[feature_columns]
y = model_data["sales"]

# Keep the final six months for a chronological test set.
split_at = len(model_data) - 6
X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2,
)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
print("Future Sales Prediction")
print("-----------------------")
print(f"Test MAE: {mean_absolute_error(y_test, test_predictions):.2f}")

print("\nActual vs predicted test sales:")
for date, actual, predicted in zip(
    model_data["date"].iloc[split_at:],
    y_test,
    test_predictions,
):
    print(
        f"{date:%Y-%m}: actual={actual:.2f}, "
        f"predicted={predicted:.2f}"
    )

# Refit using all observed sales, then forecast the next month.
model.fit(X, y)

last = data.iloc[-1]
next_month_date = last["date"] + pd.offsets.MonthBegin(1)
recent_sales = data["sales"].iloc[-3:].to_numpy()

next_month_features = pd.DataFrame([{
    "month": next_month_date.month,
    "time_index": len(data),
    "lag_1": recent_sales[-1],
    "lag_2": recent_sales[-2],
    "lag_3": recent_sales[-3],
    "rolling_mean_3": recent_sales.mean(),
}])

next_month_prediction = model.predict(next_month_features)[0]

print(f"\nPredicted sales for {next_month_date:%Y-%m}: "
      f"{next_month_prediction:.2f}")

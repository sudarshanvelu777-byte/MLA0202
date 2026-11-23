
# Install once if needed:
# pip install scikit-learn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample data: [monthly income, missed payments, debt-to-income %, credit history (years)]
X = [
    [5000, 0, 20, 10],
    [4500, 1, 25, 8],
    [6000, 0, 18, 12],
    [2500, 5, 70, 1],
    [3000, 4, 65, 2],
    [2800, 6, 80, 1],
    [4000, 2, 40, 5],
    [3500, 3, 50, 4],
    [7000, 0, 15, 15],
    [2200, 7, 85, 1],
    [5200, 1, 30, 9],
    [3200, 4, 60, 3],
    [6500, 0, 22, 13],
    [3800, 2, 45, 6],
    [2700, 5, 75, 2],
]

# Labels correspond to the rows above.
y = [
    "Good", "Good", "Good",
    "Poor", "Poor", "Poor",
    "Standard", "Standard", "Good",
    "Poor", "Good", "Poor",
    "Good", "Standard", "Poor",
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Credit Score Classification Results")
print("-----------------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Predict one new sample:
new_customer = [[4200, 1, 35, 7]]
print("New customer's predicted category:", model.predict(new_customer)[0])

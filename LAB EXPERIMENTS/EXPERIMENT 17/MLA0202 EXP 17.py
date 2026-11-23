from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Generate sample data with mobile-like features:
# battery power, Bluetooth, clock speed, dual SIM, front camera,
# 4G, internal memory, mobile weight, number of cores, primary camera,
# pixel height, pixel width, RAM, screen height, screen width,
# talk time, 3G, touch screen, Wi-Fi
X, y = make_classification(
    n_samples=2000,
    n_features=19,
    n_informative=12,
    n_redundant=3,
    n_classes=4,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42,
)

price_ranges = ["Low", "Medium", "High", "Very High"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ),
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Mobile Price Range Prediction")
print("-----------------------------")
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=price_ranges,
    zero_division=0,
))

# Example feature values, in the same order listed above.
# These are illustrative inputs, not a real phone specification.
new_mobile = [[
    1500, 1, 1.8, 1, 8, 1, 32, 180, 8, 12,
    900, 1600, 3000, 15, 7, 12, 1, 1, 1
]]

predicted_class = model.predict(new_mobile)[0]
print("Example prediction:", price_ranges[predicted_class])

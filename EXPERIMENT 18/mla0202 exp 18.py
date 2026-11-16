from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data while keeping class proportions similar
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Scale the measurements, then train the Perceptron
model = make_pipeline(
    StandardScaler(),
    Perceptron(max_iter=1000, tol=1e-3, random_state=42),
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Perceptron-based Iris Classification")
print("------------------------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names,
    zero_division=0,
))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new flower: sepal length, sepal width, petal length, petal width (cm)
new_flower = [[5.1, 3.5, 1.4, 0.2]]
predicted_class = model.predict(new_flower)[0]

print("\nNew flower prediction:")
print(f"Measurements: {new_flower[0]}")
print(f"Predicted species: {iris.target_names[predicted_class]}")

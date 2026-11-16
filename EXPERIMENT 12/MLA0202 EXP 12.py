# Install once if needed:
# pip install scikit-learn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the built-in Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train the KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Iris Flower Classification using KNN")
print("------------------------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new flower.
# Values are: sepal length, sepal width, petal length, petal width (cm)
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)[0]

print("\nNew flower prediction:")
print(f"Measurements: {new_flower[0]}")
print(f"Predicted species: {iris.target_names[prediction]}")

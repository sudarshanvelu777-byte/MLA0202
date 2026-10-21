# 2. K-NEAREST NEIGHBOURS (K-NN)

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train K-NN model
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
model.fit(X_train, y_train)

# Predict test samples
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("K-NEAREST NEIGHBOURS (K-NN)")
print("K value:", 5)
print("Correct predictions:", sum(y_test == y_pred))
print("Total test samples:", len(y_test))
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=data.target_names,
    cmap="Greens"
)
plt.title("K-NN Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot accuracy
plt.figure(figsize=(5, 4))
plt.bar(["K-NN (K=5)"], [accuracy * 100])
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("K-NN Classification Accuracy")
plt.tight_layout()
plt.show()

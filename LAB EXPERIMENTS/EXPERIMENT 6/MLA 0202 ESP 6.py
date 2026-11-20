# 3. NAIVE BAYES CLASSIFICATION

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict test samples
y_pred = model.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display results
print("NAIVE BAYES CLASSIFICATION")
print("Correct predictions:", sum(y_test == y_pred))
print("Total test samples:", len(y_test))
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=data.target_names
))

# Plot confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
).plot(cmap="Oranges")

plt.title("Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot accuracy
plt.figure(figsize=(5, 4))
plt.bar(["Naive Bayes"], [accuracy * 100])
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Naive Bayes Classification Accuracy")
plt.tight_layout()
plt.show()

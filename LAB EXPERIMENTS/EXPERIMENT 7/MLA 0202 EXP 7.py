# Artificial Neural Network using Backpropagation

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline

# Step 1: Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Create the ANN model
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
)

# Step 4: Train the model using backpropagation
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("ARTIFICIAL NEURAL NETWORK")
print("Correct predictions:", sum(y_test == y_pred))
print("Total test samples:", len(y_test))
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Access the trained ANN correctly
ann = model.named_steps["mlpclassifier"]

# Step 8: Plot the training loss
plt.figure(figsize=(7, 4))
plt.plot(ann.loss_curve_)
plt.title("ANN Backpropagation - Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 9: Plot the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=data.target_names,
    cmap="Blues"
)
plt.title("ANN Confusion Matrix")
plt.tight_layout()
plt.show()

# Step 10: Plot the accuracy
plt.figure(figsize=(5, 4))
plt.bar(["ANN"], [accuracy * 100])
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("ANN Classification Accuracy")
plt.tight_layout()
plt.show()

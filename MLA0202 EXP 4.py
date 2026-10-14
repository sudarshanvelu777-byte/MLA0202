import numpy as np
import matplotlib.pyplot as plt

# 1. Prepare the XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# 2. Initialize network parameters
np.random.seed(42)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1
learning_rate = 0.5
epochs = 10000

W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# 3. Train the ANN using backpropagation
loss_history = []

for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Calculate mean squared error
    loss = np.mean((y - a2) ** 2)
    loss_history.append(loss)

    # Backpropagation
    error_output = (a2 - y) * a2 * (1 - a2)
    error_hidden = np.dot(error_output, W2.T) * a1 * (1 - a1)

    # Update weights and biases
    W2 -= learning_rate * np.dot(a1.T, error_output) / len(X)
    b2 -= learning_rate * np.mean(error_output, axis=0, keepdims=True)

    W1 -= learning_rate * np.dot(X.T, error_hidden) / len(X)
    b1 -= learning_rate * np.mean(error_hidden, axis=0, keepdims=True)

# 4. Test the trained network
z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)
a2 = sigmoid(np.dot(a1, W2) + b2)

predictions = (a2 >= 0.5).astype(int)

print("ANN Results using Backpropagation")
print("---------------------------------")
for i in range(len(X)):
    print(
        f"Input: {X[i]} | "
        f"Actual: {y[i][0]} | "
        f"Predicted: {predictions[i][0]} | "
        f"Output: {a2[i][0]:.4f}"
    )

accuracy = np.mean(predictions == y) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}%")

# 5. Plot the training loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("ANN Training Loss using Backpropagation")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Visualize predicted versus actual outputs
plt.figure(figsize=(8, 5))
positions = np.arange(len(X))
width = 0.35

plt.bar(positions - width / 2, y.flatten(),
        width, label="Actual")
plt.bar(positions + width / 2, predictions.flatten(),
        width, label="Predicted")

plt.xticks(positions, ["0 XOR 0", "0 XOR 1",
                       "1 XOR 0", "1 XOR 1"])
plt.ylabel("Output")
plt.title("Actual vs Predicted XOR Outputs")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Step 1: Create a sample dataset
data = {
    'Outlook': [
        'Sunny', 'Sunny', 'Overcast', 'Rain',
        'Rain', 'Rain', 'Overcast', 'Sunny',
        'Sunny', 'Rain', 'Sunny', 'Overcast',
        'Overcast', 'Rain'
    ],
    'Temperature': [
        'Hot', 'Hot', 'Hot', 'Mild',
        'Cool', 'Cool', 'Cool', 'Mild',
        'Cool', 'Mild', 'Mild', 'Mild',
        'Hot', 'Mild'
    ],
    'Humidity': [
        'High', 'High', 'High', 'High',
        'Normal', 'Normal', 'Normal', 'High',
        'Normal', 'Normal', 'Normal', 'High',
        'Normal', 'High'
    ],
    'Wind': [
        'Weak', 'Strong', 'Weak', 'Weak',
        'Weak', 'Strong', 'Strong', 'Weak',
        'Weak', 'Weak', 'Strong', 'Strong',
        'Weak', 'Strong'
    ],
    'Play': [
        'No', 'No', 'Yes', 'Yes',
        'Yes', 'No', 'Yes', 'No',
        'Yes', 'Yes', 'Yes', 'Yes',
        'Yes', 'No'
    ]
}

df = pd.DataFrame(data)

print("Training Dataset:")
print(df)

# Step 2: Encode categorical values into numbers
encoders = {}

for column in df.columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

X = df.drop('Play', axis=1)
y = df['Play']

# Step 3: Build the ID3 decision tree
# Entropy criterion + information gain
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=0
)

model.fit(X, y)

# Step 4: Display the decision tree
plt.figure(figsize=(14, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=encoders['Play'].classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("ID3 Decision Tree - Play Tennis")
plt.tight_layout()
plt.show()

# Step 5: Classify a new sample
new_sample = pd.DataFrame([{
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'Normal',
    'Wind': 'Weak'
}])

# Encode the new sample using the training encoders
for column in new_sample.columns:
    new_sample[column] = encoders[column].transform(
        new_sample[column]
    )

prediction = model.predict(new_sample)[0]
result = encoders['Play'].inverse_transform([prediction])[0]

print("\nNew Sample:")
print({
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'Normal',
    'Wind': 'Weak'
})

print("Predicted Result:", result)

# Step 6: Calculate training accuracy
y_pred = model.predict(X)
print("Training Accuracy:", accuracy_score(y, y_pred) * 100, "%")

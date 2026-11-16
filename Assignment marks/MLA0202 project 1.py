import pandas as pd
import matplotlib.pyplot as plt

# Training dataset
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Same'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

print("Training Dataset:")
print(df.to_string(index=False))

# FIND-S algorithm
hypothesis = None
history = []

for i, row in df.iterrows():
    if row['EnjoySport'] == 'Yes':
        attributes = row.iloc[:-1].tolist()

        if hypothesis is None:
            hypothesis = attributes.copy()
        else:
            for j in range(len(hypothesis)):
                if hypothesis[j] != attributes[j]:
                    hypothesis[j] = '?'

        history.append((i + 1, hypothesis.copy()))
        print(f"After example {i + 1}: {hypothesis}")

print("\nFinal Hypothesis:", hypothesis)

# Display training data
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].axis('off')
table = axes[0].table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.6)
axes[0].set_title("Training Dataset")

# Visualize hypothesis changes
attributes = df.columns[:-1]
steps = [f"Example {item[0]}" for item in history]
values = [item[1] for item in history]

for j, attribute in enumerate(attributes):
    axes[1].plot(
        steps,
        [str(h[j]) for h in values],
        marker='o',
        label=attribute
    )

axes[1].set_title("FIND-S Hypothesis Evolution")
axes[1].set_xlabel("Positive Training Examples")
axes[1].set_ylabel("Hypothesis Attribute Values")
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

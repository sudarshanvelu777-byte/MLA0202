import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Load the dataset included with scikit-learn
iris = load_iris()
X = iris.data
y = iris.target

# Scaling is included in the pipelines so it is fitted separately
# within each cross-validation fold.
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    ),
    "K-Nearest Neighbors": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    ),
    "Support Vector Machine": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf"),
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
    "Gaussian Naive Bayes": GaussianNB(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        error_score="raise",
    )

    results.append({
        "Algorithm": name,
        "Accuracy (%)": scores["test_accuracy"].mean() * 100,
        "Precision (%)": scores["test_precision_weighted"].mean() * 100,
        "Recall (%)": scores["test_recall_weighted"].mean() * 100,
        "F1 Score (%)": scores["test_f1_weighted"].mean() * 100,
    })

results_df = pd.DataFrame(results).sort_values(
    by="Accuracy (%)",
    ascending=False,
)

print("Classifier Performance on the Iris Dataset (5-fold cross-validation)")
print(results_df.to_string(
    index=False,
    formatters={
        "Accuracy (%)": "{:.2f}".format,
        "Precision (%)": "{:.2f}".format,
        "Recall (%)": "{:.2f}".format,
        "F1 Score (%)": "{:.2f}".format,
    },
))

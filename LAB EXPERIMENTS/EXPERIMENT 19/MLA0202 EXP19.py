import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Fictional sample loan application data
data = pd.DataFrame({
    "income": [35000, 52000, 28000, 75000, 42000, 90000, 31000, 62000,
               47000, 25000, 80000, 39000, 56000, 33000, 70000, 29000,
               65000, 48000, 85000, 36000, 59000, 27000, 72000, 44000],
    "credit_score": [620, 720, 580, 780, 680, 810, 610, 740,
                     690, 550, 790, 650, 730, 600, 770, 570,
                     750, 700, 800, 630, 740, 560, 760, 670],
    "loan_amount": [15000, 20000, 12000, 25000, 18000, 30000, 14000, 22000,
                    17000, 10000, 28000, 16000, 21000, 13000, 24000, 11000,
                    23000, 19000, 29000, 15000, 22000, 9000, 26000, 18000],
    "employment_years": [2, 8, 1, 12, 5, 15, 2, 10, 6, 1, 14, 4,
                         9, 2, 11, 1, 10, 7, 16, 3, 9, 1, 13, 5],
    "home_owner": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes",
                   "No", "No", "Yes", "No", "Yes", "No", "Yes", "No",
                   "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    "loan_approved": [
        "No", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes",
        "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No",
        "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes"
    ],
})

target = "loan_approved"
X = data.drop(columns=[target])
y = data[target]

numeric_columns = ["income", "credit_score", "loan_amount", "employment_years"]
categorical_columns = ["home_owner"]

# OneHotEncoder compatibility with older and newer scikit-learn versions
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_columns),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ]), categorical_columns),
])

# Gaussian Naive Bayes is suitable for numeric feature values.
# The preprocessor converts the categorical feature into numeric columns.
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", GaussianNB()),
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Bank Loan Approval Prediction using Naive Bayes")
print("------------------------------------------------")
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=["No", "Yes"]))

# Predict a new loan application
new_applicant = pd.DataFrame([{
    "income": 60000,
    "credit_score": 740,
    "loan_amount": 20000,
    "employment_years": 7,
    "home_owner": "Yes",
}])

prediction = model.predict(new_applicant)[0]
print("\nExample application prediction:", prediction)

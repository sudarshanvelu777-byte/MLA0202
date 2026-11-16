
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Credit_Score"
DROP_COLUMNS = ["ID", "Customer_ID", "Name", "SSN", "Month"]


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
            continue

        values = df[col].astype("string").str.strip()
        values = values.replace(
            ["", "NA", "N/A", "nan", "None", "!@9#%8", "_______", "_"],
            pd.NA,
        )

        if col == "Credit_History_Age":
            years = pd.to_numeric(
                values.str.extract(r"(\d+)\s*Years?", expand=False),
                errors="coerce",
            ).fillna(0)
            months = pd.to_numeric(
                values.str.extract(r"(\d+)\s*Months?", expand=False),
                errors="coerce",
            ).fillna(0)
            df[col] = years * 12 + months
            df.loc[values.isna(), col] = float("nan")
            continue

        cleaned = (
            values.str.replace(",", "", regex=False)
            .str.replace("_", "", regex=False)
            .str.replace(r"^\$", "", regex=True)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")

        valid_count = values.notna().sum()
        numeric_count = numeric.notna().sum()

        if valid_count > 0 and numeric_count / valid_count >= 0.5:
            df[col] = numeric
        else:
            df[col] = values.str.strip("_").replace("", pd.NA)

    return df


def choose_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def main():
    print("Select your TRAINING CSV file.")
    train_file = choose_file(
        "Select training CSV file",
        [("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not train_file:
        print("No training file selected. Program stopped.")
        return

    train_df = pd.read_csv(train_file, low_memory=False)
    train_df.columns = train_df.columns.str.strip()

    if TARGET not in train_df.columns:
        raise ValueError(
            f"Could not find the '{TARGET}' column.\n"
            f"Columns found: {train_df.columns.tolist()}\n"
            "Select the labeled training CSV, not the test CSV."
        )

    train_df[TARGET] = train_df[TARGET].astype("string").str.strip()
    train_df = train_df.dropna(subset=[TARGET])
    train_df = train_df[train_df[TARGET] != ""]

    y = train_df[TARGET]
    X = clean_data(train_df.drop(columns=[TARGET]))
    X = X.drop(columns=DROP_COLUMNS, errors="ignore")

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns were found in the training CSV.")

    preprocessing = ColumnTransformer([
        ("numbers", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_columns),
        ("categories", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", make_one_hot_encoder()),
        ]), categorical_columns),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    print("\nValidation report:")
    print(classification_report(y_valid, predictions, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_valid, predictions))
    print("\nValidation finished.")

    answer = input(
        "\nDo you also have a separate TEST CSV file to predict? (y/n): "
    ).strip().lower()

    if answer == "y":
        print("Select your TEST CSV file.")
        test_file = choose_file(
            "Select test CSV file",
            [("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if test_file:
            test_df = pd.read_csv(test_file, low_memory=False)
            test_df.columns = test_df.columns.str.strip()

            ids = (
                test_df["ID"]
                if "ID" in test_df.columns
                else pd.Series(range(len(test_df)), name="row")
            )

            test_X = clean_data(test_df).drop(
                columns=DROP_COLUMNS, errors="ignore"
            )

            model.fit(X, y)
            test_predictions = model.predict(test_X)

            output_file = Path(test_file).with_name(
                "credit_score_predictions.csv"
            )
            pd.DataFrame({
                "ID": ids,
                TARGET: test_predictions,
            }).to_csv(output_file, index=False)

            print(f"Predictions saved here:\n{output_file}")
        else:
            print("No test file selected; no predictions were created.")

    print(f"\nscikit-learn version: {sklearn.__version__}")


if __name__ == "__main__":
    main()
    

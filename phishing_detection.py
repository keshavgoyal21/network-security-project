"""
Phishing Detection Project Using Real PhiUSIIL Dataset
======================================================

This project uses a labelled phishing URL dataset for a binary classification
task. The original PhiUSIIL dataset label values are:

- 1 = legitimate / safe
- 0 = phishing

For this project, labels are converted to the common project format:

- 0 = safe
- 1 = phishing

How to run:

1. Put `PhiUSIIL_Phishing_URL_Dataset.csv` in the same folder as this script.
2. Install dependencies:
   pip install pandas numpy scikit-learn
3. Run:
   python phishing_real_dataset_project.py

Generated file:

- No output files are generated. Tables print on screen and diagrams display on screen.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_FILE = "PhiUSIIL_Phishing_URL_Dataset.csv"
TARGET_COLUMN = "label"
PROJECT_TARGET_COLUMN = "phishing_label"

TEST_SIZE = 0.2
RANDOM_STATE = 42
RANDOM_FOREST_TREES = 100
LOGISTIC_REGRESSION_MAX_ITER = 1000
TOP_FEATURE_COUNT = 20

ORIGINAL_SAFE_LABEL = 1
ORIGINAL_PHISHING_LABEL = 0
PROJECT_SAFE_LABEL = 0
PROJECT_PHISHING_LABEL = 1

TEXT_COLUMNS_TO_DROP = ["FILENAME", "URL", "Domain", "Title"]
CATEGORICAL_COLUMNS = ["TLD"]

LABEL_NAMES = ["Safe", "Phishing"]


def make_heading(title: str) -> str:
    """Create a clean heading for report-style output."""
    border = "=" * len(title)
    return f"\n{border}\n{title}\n{border}"


def format_table(dataframe: pd.DataFrame, max_rows: int | None = None) -> str:
    """Convert a DataFrame into a readable aligned table."""
    if max_rows is not None:
        dataframe = dataframe.head(max_rows)
    return dataframe.to_string(index=False)


def print_and_store(lines: list[str], text: str = "") -> None:
    """Print text to screen and also store it for the results file."""
    print(text)
    lines.append(text)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the CSV dataset into a Pandas DataFrame."""
    return pd.read_csv(file_path, low_memory=False)


def get_dataset_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create a table containing column names, data types, and non-null counts."""
    summary = pd.DataFrame(
        {
            "Column": dataset.columns,
            "Data Type": dataset.dtypes.astype(str).values,
            "Non-Null Count": dataset.notna().sum().values,
            "Missing Count": dataset.isna().sum().values,
        }
    )
    return summary


def get_missing_values_table(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create a table showing missing values for every column."""
    missing_count = dataset.isna().sum()
    missing_percentage = (missing_count / len(dataset)) * 100
    missing_table = pd.DataFrame(
        {
            "Column": missing_count.index,
            "Missing Values": missing_count.values,
            "Missing Percentage": missing_percentage.round(4).values,
        }
    )
    return missing_table


def identify_features_and_target(dataset: pd.DataFrame) -> tuple[list[str], str]:
    """Identify independent feature columns and dependent target column."""
    feature_columns = [column for column in dataset.columns if column != TARGET_COLUMN]
    return feature_columns, TARGET_COLUMN


def convert_target_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    """Convert original dataset labels into project labels: 0 safe, 1 phishing."""
    converted_dataset = dataset.copy()
    converted_dataset[PROJECT_TARGET_COLUMN] = converted_dataset[TARGET_COLUMN].map(
        {
            ORIGINAL_SAFE_LABEL: PROJECT_SAFE_LABEL,
            ORIGINAL_PHISHING_LABEL: PROJECT_PHISHING_LABEL,
        }
    )
    return converted_dataset


def preprocess_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Handle missing values, encode categorical columns, and prepare X and y."""
    processed_dataset = convert_target_labels(dataset)
    y = processed_dataset[PROJECT_TARGET_COLUMN].astype(int)

    columns_to_drop = [TARGET_COLUMN, PROJECT_TARGET_COLUMN]
    columns_to_drop.extend(
        column for column in TEXT_COLUMNS_TO_DROP if column in processed_dataset.columns
    )

    x = processed_dataset.drop(columns=columns_to_drop)

    numeric_columns = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in CATEGORICAL_COLUMNS if column in x.columns
    ]

    numeric_features = x[numeric_columns].copy()
    categorical_features = x[categorical_columns].copy()

    for column in numeric_features.columns:
        median_value = numeric_features[column].median()
        numeric_features[column] = numeric_features[column].fillna(median_value)

    for column in categorical_features.columns:
        mode_values = categorical_features[column].mode()
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        categorical_features[column] = categorical_features[column].fillna(fill_value)

    encoded_categorical = pd.get_dummies(
        categorical_features,
        columns=categorical_columns,
        drop_first=True,
        dtype=int,
    )

    x_processed = pd.concat([numeric_features, encoded_categorical], axis=1)
    return x_processed, y


def get_split_table(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Create a table showing train-test split information."""
    total_rows = len(x_train) + len(x_test)
    split_table = pd.DataFrame(
        {
            "Dataset Part": ["Training Set", "Testing Set", "Total Dataset"],
            "Rows": [len(x_train), len(x_test), total_rows],
            "Percentage": [
                round((len(x_train) / total_rows) * 100, 2),
                round((len(x_test) / total_rows) * 100, 2),
                100.00,
            ],
            "Features": [x_train.shape[1], x_test.shape[1], x_train.shape[1]],
            "Target Rows": [len(y_train), len(y_test), len(y_train) + len(y_test)],
        }
    )
    return split_table


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[LogisticRegression, StandardScaler]:
    """Scale features and train a Logistic Regression model."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LogisticRegression(
        max_iter=LOGISTIC_REGRESSION_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train_scaled, y_train)
    return model, scaler


def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train a Random Forest classification model."""
    model = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model_name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float | str]:
    """Calculate classification metrics for one model."""
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }


def build_metrics_table(metrics: list[dict[str, float | str]]) -> pd.DataFrame:
    """Create a formatted model comparison metrics table."""
    metrics_table = pd.DataFrame(metrics)
    numeric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metrics_table[numeric_columns] = metrics_table[numeric_columns].round(4)
    return metrics_table


def build_confusion_matrix_table(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Create confusion matrix as a formatted table."""
    matrix = confusion_matrix(y_test, y_pred, labels=[PROJECT_SAFE_LABEL, PROJECT_PHISHING_LABEL])
    matrix_table = pd.DataFrame(
        matrix,
        index=["Actual Safe", "Actual Phishing"],
        columns=["Predicted Safe", "Predicted Phishing"],
    ).reset_index(names="Actual / Predicted")
    matrix_table.insert(0, "Model", model_name)
    return matrix_table


def build_feature_importance_table(
    model: RandomForestClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """Create a feature importance table from the Random Forest model."""
    importance_table = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance Score": model.feature_importances_,
        }
    )
    importance_table = importance_table.sort_values(
        "Importance Score",
        ascending=False,
    )
    importance_table["Importance Score"] = importance_table["Importance Score"].round(6)
    return importance_table


def show_confusion_matrix_diagram(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Display the confusion matrix diagram on screen without saving it."""
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[PROJECT_SAFE_LABEL, PROJECT_PHISHING_LABEL],
    )

    plt.figure(figsize=(7, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], LABEL_NAMES)
    plt.yticks([0, 1], LABEL_NAMES)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(
                column_index,
                row_index,
                matrix[row_index, column_index],
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def show_feature_importance_diagram(
    feature_importance_table: pd.DataFrame,
) -> None:
    """Display top Random Forest feature importances on screen without saving them."""
    top_features = feature_importance_table.head(TOP_FEATURE_COUNT).copy()

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["Feature"],
        top_features["Importance Score"],
    )
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def run_project() -> None:
    """Run the complete real-dataset phishing classification project."""
    report_lines: list[str] = []

    print_and_store(report_lines, make_heading("PHISHING DETECTION PROJECT USING REAL LABELLED DATASET"))

    dataset = load_dataset(DATASET_FILE)

    print_and_store(report_lines, make_heading("STEP 1: FIRST 5 ROWS OF DATASET"))
    display_columns = [
        "URL",
        "URLLength",
        "Domain",
        "DomainLength",
        "TLD",
        "IsHTTPS",
        "label",
    ]
    available_display_columns = [
        column for column in display_columns if column in dataset.columns
    ]
    print_and_store(
        report_lines,
        format_table(dataset[available_display_columns].head(5)),
    )

    print_and_store(report_lines, make_heading("STEP 2: FEATURES AND TARGET VARIABLE"))
    feature_columns, target_column = identify_features_and_target(dataset)
    features_target_table = pd.DataFrame(
        {
            "Item": ["Independent Variables / Features", "Dependent Variable / Target"],
            "Columns": [", ".join(feature_columns[:12]) + " ...", target_column],
            "Meaning": [
                "URL and website properties used for prediction",
                "Original class label from dataset",
            ],
        }
    )
    print_and_store(report_lines, format_table(features_target_table))

    print_and_store(report_lines, make_heading("STEP 3: DATASET SUMMARY"))
    print_and_store(report_lines, format_table(get_dataset_summary(dataset)))

    print_and_store(report_lines, make_heading("STEP 4: MISSING VALUES TABLE"))
    print_and_store(report_lines, format_table(get_missing_values_table(dataset)))

    print_and_store(report_lines, make_heading("STEP 5: DATA PREPROCESSING AND ENCODING"))
    x, y = preprocess_dataset(dataset)
    preprocessing_table = pd.DataFrame(
        {
            "Preprocessing Task": [
                "Missing numeric values",
                "Missing categorical values",
                "Categorical encoding",
                "Dropped text identifier columns",
                "Target conversion",
                "Feature scaling",
            ],
            "Action Taken": [
                "Filled using median",
                "Filled using mode",
                "TLD encoded using one-hot encoding",
                ", ".join(TEXT_COLUMNS_TO_DROP),
                "Original label converted to 0 = Safe, 1 = Phishing",
                "Applied only for Logistic Regression",
            ],
        }
    )
    print_and_store(report_lines, format_table(preprocessing_table))

    print_and_store(report_lines, make_heading("STEP 6: ENCODED DATASET PREVIEW"))
    preview_columns = x.columns[:15].tolist()
    encoded_preview = x[preview_columns].head(5).copy()
    encoded_preview[PROJECT_TARGET_COLUMN] = y.head(5).values
    print_and_store(report_lines, format_table(encoded_preview, max_rows=5))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print_and_store(report_lines, make_heading("STEP 7: TRAIN-TEST SPLIT"))
    print_and_store(report_lines, format_table(get_split_table(x_train, x_test, y_train, y_test)))

    print_and_store(report_lines, make_heading("STEP 8: MODEL TRAINING"))
    logistic_model, scaler = train_logistic_regression(x_train, y_train)
    random_forest_model = train_random_forest(x_train, y_train)
    training_table = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest"],
            "Purpose": [
                "Linear baseline classification model",
                "Tree-based ensemble classification model",
            ],
            "Scaling Used": ["Yes", "No"],
        }
    )
    print_and_store(report_lines, format_table(training_table))

    logistic_predictions = logistic_model.predict(scaler.transform(x_test))
    random_forest_predictions = random_forest_model.predict(x_test)

    metrics = [
        evaluate_model("Logistic Regression", y_test, logistic_predictions),
        evaluate_model("Random Forest", y_test, random_forest_predictions),
    ]
    metrics_table = build_metrics_table(metrics)

    print_and_store(report_lines, make_heading("STEP 9: MODEL EVALUATION METRICS"))
    print_and_store(report_lines, format_table(metrics_table))

    best_model_name = metrics_table.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
    best_predictions = (
        random_forest_predictions
        if best_model_name == "Random Forest"
        else logistic_predictions
    )

    print_and_store(report_lines, make_heading("STEP 10: CONFUSION MATRIX TABLES"))
    confusion_matrix_tables = pd.concat(
        [
            build_confusion_matrix_table(
                y_test,
                logistic_predictions,
                "Logistic Regression",
            ),
            build_confusion_matrix_table(
                y_test,
                random_forest_predictions,
                "Random Forest",
            ),
        ],
        ignore_index=True,
    )
    print_and_store(report_lines, format_table(confusion_matrix_tables))

    print_and_store(report_lines, make_heading("STEP 10A: CONFUSION MATRIX MEANING"))
    matrix_meaning_table = pd.DataFrame(
        {
            "Term": [
                "Predicted Safe",
                "Predicted Phishing",
                "Actual Safe",
                "Actual Phishing",
            ],
            "Meaning": [
                "Model predicted the URL as safe",
                "Model predicted the URL as phishing",
                "URL is truly safe in the dataset",
                "URL is truly phishing in the dataset",
            ],
        }
    )
    print_and_store(report_lines, format_table(matrix_meaning_table))

    print_and_store(report_lines, make_heading("STEP 11: MODEL COMPARISON TABLE"))
    comparison_table = metrics_table.copy()
    comparison_table["Rank By F1 Score"] = (
        comparison_table["F1 Score"].rank(ascending=False, method="dense").astype(int)
    )
    comparison_table = comparison_table.sort_values("Rank By F1 Score")
    print_and_store(report_lines, format_table(comparison_table))

    print_and_store(report_lines, make_heading("STEP 12: RANDOM FOREST FEATURE IMPORTANCE"))
    feature_importance_table = build_feature_importance_table(
        random_forest_model,
        x.columns.tolist(),
    )
    print_and_store(report_lines, format_table(feature_importance_table.head(TOP_FEATURE_COUNT)))

    show_confusion_matrix_diagram(y_test, best_predictions, best_model_name)
    show_feature_importance_diagram(feature_importance_table)

    print_and_store(report_lines, make_heading("STEP 13: DIAGRAM DISPLAY STATUS"))
    diagram_table = pd.DataFrame(
        {
            "Diagram": [
                "Confusion Matrix",
                "Random Forest Feature Importance",
            ],
            "Output Method": [
                "Displayed on screen only",
                "Displayed on screen only",
            ],
            "Saved In Folder": [
                "No",
                "No",
            ],
        }
    )
    print_and_store(report_lines, format_table(diagram_table))

    print_and_store(report_lines, make_heading("FINAL RESULT"))
    final_result_table = pd.DataFrame(
        {
            "Final Output": [
                "Best Model",
                "Target Format",
                "Dataset Type",
                "Total Rows",
                "Total Features After Encoding",
            ],
            "Value": [
                best_model_name,
                "0 = Safe, 1 = Phishing",
                "Labelled classification dataset",
                len(dataset),
                x.shape[1],
            ],
        }
    )
    print_and_store(report_lines, format_table(final_result_table))


if __name__ == "__main__":
    run_project()

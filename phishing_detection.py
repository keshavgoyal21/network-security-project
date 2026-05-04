"""
Phishing Detection System
=========================

Classifies URLs as Safe or Phishing using 5 URL-based features selected from
the PhiUSIIL labelled dataset.

The dataset contains precomputed phishing-related URL features. Only
assignment-relevant URL-based features were selected. The required features
(URL length, HTTPS presence, and special character count) are used along with
two additional URL-based indicators: subdomain count and IP-domain detection.

Label mapping (original dataset):
    1 = legitimate / safe  →  project label 0 = Safe
    0 = phishing           →  project label 1 = Phishing

How to run:
    1. Place PhiUSIIL_Phishing_URL_Dataset.csv in the same folder as this script.
    2. pip install pandas numpy scikit-learn matplotlib
    3. python phishing_detection.py
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
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DATASET_FILE = "PhiUSIIL_Phishing_URL_Dataset.csv"

# Exactly the 5 features required by the assignment
SELECTED_FEATURES = [
    "URLLength",                   # Required: URL length
    "IsHTTPS",                     # Required: HTTPS presence (1 = yes, 0 = no)
    "NoOfOtherSpecialCharsInURL",  # Required: special character count
    "NoOfSubDomain",               # Extra: subdomain count
    "IsDomainIP",                  # Extra: IP-based domain flag
]

TARGET_COLUMN = "label"           # Original dataset label column

# Original dataset encoding  →  project encoding
ORIGINAL_SAFE_LABEL     = 1       # dataset: 1 = legitimate
ORIGINAL_PHISHING_LABEL = 0       # dataset: 0 = phishing
PROJECT_SAFE_LABEL      = 0       # project: 0 = Safe
PROJECT_PHISHING_LABEL  = 1       # project: 1 = Phishing

LABEL_NAMES   = ["Safe", "Phishing"]
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
RF_TREES      = 100
LR_MAX_ITER   = 1000


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────

def heading(title: str) -> str:
    border = "=" * len(title)
    return f"\n{border}\n{title}\n{border}"


def fmt(df: pd.DataFrame, rows: int | None = None) -> str:
    return (df.head(rows) if rows else df).to_string(index=False)


# ──────────────────────────────────────────────
# Data loading & preparation
# ──────────────────────────────────────────────

def load_and_prepare(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the CSV, select the 5 features, convert labels, drop any rows with
    missing values in the selected columns.
    """
    raw = pd.read_csv(file_path, low_memory=False)

    # Keep only the 5 chosen features + label
    cols_needed = SELECTED_FEATURES + [TARGET_COLUMN]
    df = raw[cols_needed].copy()

    # Convert original labels → project labels
    df["phishing_label"] = df[TARGET_COLUMN].map(
        {
            ORIGINAL_SAFE_LABEL:     PROJECT_SAFE_LABEL,
            ORIGINAL_PHISHING_LABEL: PROJECT_PHISHING_LABEL,
        }
    )

    # Drop rows where any selected feature or label is missing
    df.dropna(subset=SELECTED_FEATURES + ["phishing_label"], inplace=True)

    X = df[SELECTED_FEATURES].astype(float)
    y = df["phishing_label"].astype(int)

    return X, y


# ──────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────

def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=RF_TREES, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred) -> dict:
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def confusion_table(y_true, y_pred, name: str) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        {
            "Model":              [name] * 4,
            "Actual":             ["Safe", "Safe", "Phishing", "Phishing"],
            "Predicted":          ["Safe", "Phishing", "Safe", "Phishing"],
            "Count":              [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]],
            "Description":        [
                "True Negative  (TN)",
                "False Positive (FP)",
                "False Negative (FN)",
                "True Positive  (TP)",
            ],
        }
    )


def show_confusion_matrix(y_true, y_pred, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        xlabel="Predicted Label", ylabel="True Label",
        title=f"Confusion Matrix – {title}",
    )
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()


def show_feature_importance(model: RandomForestClassifier, features: list[str]) -> None:
    importances = model.feature_importances_
    fi = pd.DataFrame({"Feature": features, "Importance": importances})
    fi.sort_values("Importance", ascending=True, inplace=True)

    plt.figure(figsize=(7, 4))
    plt.barh(fi["Feature"], fi["Importance"], color="steelblue")
    plt.xlabel("Importance Score")
    plt.title("Random Forest – Feature Importance")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Sample predictions using manual URL properties
# ──────────────────────────────────────────────

SAMPLE_URLS = [
    {
        "url":         "https://google.com",
        "URLLength":                  18,
        "IsHTTPS":                     1,
        "NoOfOtherSpecialCharsInURL":  0,
        "NoOfSubDomain":               0,
        "IsDomainIP":                  0,
    },
    {
        "url":         "http://paypal-login-security-update.xyz/verify?user=abc&token=xyz",
        "URLLength":                  63,
        "IsHTTPS":                     0,
        "NoOfOtherSpecialCharsInURL":  5,
        "NoOfSubDomain":               1,
        "IsDomainIP":                  0,
    },
    {
        "url":         "http://192.168.1.1/admin/login",
        "URLLength":                  30,
        "IsHTTPS":                     0,
        "NoOfOtherSpecialCharsInURL":  2,
        "NoOfSubDomain":               0,
        "IsDomainIP":                  1,
    },
    {
        "url":         "https://mail.google.com/mail/u/0/#inbox",
        "URLLength":                  39,
        "IsHTTPS":                     1,
        "NoOfOtherSpecialCharsInURL":  2,
        "NoOfSubDomain":               1,
        "IsDomainIP":                  0,
    },
    {
        "url":         "http://free-iphone-winner.xyz/claim?ref=abc&promo=1&id=99",
        "URLLength":                  57,
        "IsHTTPS":                     0,
        "NoOfOtherSpecialCharsInURL":  6,
        "NoOfSubDomain":               1,
        "IsDomainIP":                  0,
    },
]


def show_sample_predictions(model, scaler, sample_urls: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in sample_urls:
        # Use a DataFrame with proper column names to avoid sklearn feature-name warning
        feats_df = pd.DataFrame(
            [[
                entry["URLLength"],
                entry["IsHTTPS"],
                entry["NoOfOtherSpecialCharsInURL"],
                entry["NoOfSubDomain"],
                entry["IsDomainIP"],
            ]],
            columns=SELECTED_FEATURES,
        )
        feats_scaled = scaler.transform(feats_df)
        pred = model.predict(feats_scaled)[0]
        rows.append({
            "URL":        entry["url"],
            "Prediction": LABEL_NAMES[pred],
        })
    return pd.DataFrame(rows)


def show_train_vs_test(lr_model, scaler, rf_model, X_train, y_train, X_test, y_test) -> None:
    """Combined bar chart comparing train vs test accuracy for both models."""
    lr_train_acc = accuracy_score(y_train, lr_model.predict(scaler.transform(X_train)))
    lr_test_acc  = accuracy_score(y_test,  lr_model.predict(scaler.transform(X_test)))
    rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
    rf_test_acc  = accuracy_score(y_test,  rf_model.predict(X_test))

    models = ["Logistic Regression", "Random Forest"]
    train_scores = [lr_train_acc, rf_train_acc]
    test_scores  = [lr_test_acc,  rf_test_acc]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, train_scores, width, label="Train Accuracy", color="steelblue")
    bars2 = ax.bar(x + width / 2, test_scores,  width, label="Test Accuracy",  color="coral")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Test Accuracy – Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    for bar in list(bars1) + list(bars2):
        ax.annotate(
            f"{bar.get_height():.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    plt.show()


def show_roc_curves(lr_model, scaler, rf_model, X_test, y_test) -> None:
    """ROC curve for both models on one plot."""
    lr_probs = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    rf_auc = roc_auc_score(y_test, rf_probs)

    plt.figure(figsize=(7, 5))
    plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression  (AUC = {lr_auc:.4f})", color="steelblue")
    plt.plot(rf_fpr, rf_tpr, label=f"Random Forest        (AUC = {rf_auc:.4f})", color="coral")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def run_project() -> None:
    print(heading("PHISHING DETECTION SYSTEM"))
    print("Features used:", SELECTED_FEATURES)

    # ── Step 1: Load dataset ──────────────────
    print(heading("STEP 1: LOAD DATASET"))
    X, y = load_and_prepare(DATASET_FILE)
    print(f"Total samples loaded : {len(X):,}")
    print(f"Safe (0)             : {(y == 0).sum():,}")
    print(f"Phishing (1)         : {(y == 1).sum():,}")

    # ── Step 2: Feature preview ───────────────
    print(heading("STEP 2: SELECTED FEATURES PREVIEW (first 5 rows)"))
    preview = X.head(5).copy()
    preview["Label"] = y.head(5).values
    print(fmt(preview))

    feature_info = pd.DataFrame(
        {
            "Feature":     SELECTED_FEATURES,
            "Type":        ["Required", "Required", "Required", "Extra", "Extra"],
            "Description": [
                "Length of the URL",
                "1 = HTTPS, 0 = HTTP",
                "Count of suspicious special characters",
                "Number of subdomains in the URL",
                "1 if domain is an IP address, else 0",
            ],
        }
    )
    print(f"\nFeature descriptions:\n{fmt(feature_info)}")

    # ── Step 3: Train-test split ──────────────
    print(heading("STEP 3: TRAIN-TEST SPLIT  (80% / 20%)"))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    split_table = pd.DataFrame(
        {
            "Partition": ["Training Set", "Testing Set"],
            "Rows":      [len(X_train), len(X_test)],
            "Pct":       [
                f"{len(X_train)/len(X)*100:.1f}%",
                f"{len(X_test)/len(X)*100:.1f}%",
            ],
        }
    )
    print(fmt(split_table))

    # ── Step 4: Train models ──────────────────
    print(heading("STEP 4: MODEL TRAINING"))
    lr_model, scaler      = train_logistic_regression(X_train, y_train)
    rf_model              = train_random_forest(X_train, y_train)
    print("✔ Logistic Regression trained")
    print("✔ Random Forest trained")

    # ── Step 5: Predictions ───────────────────
    lr_pred = lr_model.predict(scaler.transform(X_test))
    rf_pred = rf_model.predict(X_test)

    # ── Step 6: Evaluation metrics ────────────
    print(heading("STEP 5: EVALUATION METRICS"))
    metrics = pd.DataFrame([
        evaluate("Logistic Regression", y_test, lr_pred),
        evaluate("Random Forest",       y_test, rf_pred),
    ])
    print(fmt(metrics))

    # ── Step 7: Confusion matrices (text) ─────
    print(heading("STEP 6: CONFUSION MATRIX TABLES"))
    cm_tables = pd.concat(
        [
            confusion_table(y_test, lr_pred, "Logistic Regression"),
            confusion_table(y_test, rf_pred, "Random Forest"),
        ],
        ignore_index=True,
    )
    print(fmt(cm_tables))

    # ── Step 8: Model comparison ──────────────
    print(heading("STEP 7: MODEL COMPARISON"))
    comparison = metrics.copy()
    comparison["Rank (F1)"] = (
        comparison["F1 Score"].rank(ascending=False, method="dense").astype(int)
    )
    comparison.sort_values("Rank (F1)", inplace=True)
    print(fmt(comparison))

    best_name = comparison.iloc[0]["Model"]
    best_pred = rf_pred if best_name == "Random Forest" else lr_pred
    print(f"\n→ Best model: {best_name}")

    # ── Step 9: Feature importance ────────────
    print(heading("STEP 8: RANDOM FOREST FEATURE IMPORTANCE"))
    fi = pd.DataFrame(
        {
            "Feature":    SELECTED_FEATURES,
            "Importance": rf_model.feature_importances_.round(4),
        }
    ).sort_values("Importance", ascending=False)
    print(fmt(fi))

    # ── Step 10: Sample predictions ───────────
    print(heading("STEP 9: SAMPLE PREDICTIONS  (Logistic Regression)"))
    samples = show_sample_predictions(lr_model, scaler, SAMPLE_URLS)
    print(fmt(samples))
    for _, row in samples.iterrows():
        tag = "✅ Safe" if row["Prediction"] == "Safe" else "⚠️  Phishing"
        print(f"  {tag}  →  {row['URL']}")

    # ── Step 11: Final summary ────────────────
    print(heading("FINAL RESULT"))
    summary = pd.DataFrame(
        {
            "Item": [
                "Dataset",
                "Features used",
                "Total samples",
                "Train / Test split",
                "Best model",
                "Best model accuracy",
                "Best model F1 score",
                "Label encoding",
            ],
            "Value": [
                "PhiUSIIL Phishing URL Dataset",
                ", ".join(SELECTED_FEATURES),
                f"{len(X):,}",
                "80% / 20%",
                best_name,
                comparison.iloc[0]["Accuracy"],
                comparison.iloc[0]["F1 Score"],
                "0 = Safe  |  1 = Phishing",
            ],
        }
    )
    print(fmt(summary))

    # ── Visualisations ────────────────────────
    show_confusion_matrix(y_test, best_pred, best_name)
    show_feature_importance(rf_model, SELECTED_FEATURES)
    show_train_vs_test(lr_model, scaler, rf_model, X_train, y_train, X_test, y_test)
    show_roc_curves(lr_model, scaler, rf_model, X_test, y_test)


if __name__ == "__main__":
    run_project()

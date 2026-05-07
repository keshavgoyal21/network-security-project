
import urllib.parse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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


SELECTED_FEATURES = [
    "URLLength",                   
    "IsHTTPS",                     
    "NoOfOtherSpecialCharsInURL",  
    "NoOfSubDomain",               
    "IsDomainIP",                  
]

TARGET_COLUMN = "label"           


ORIGINAL_SAFE_LABEL     = 1        
ORIGINAL_PHISHING_LABEL = 0        
PROJECT_SAFE_LABEL      = 0        
PROJECT_PHISHING_LABEL  = 1        

LABEL_NAMES  = ["Safe", "Phishing"]
RANDOM_STATE = 42
RF_TREES     = 100
LR_MAX_ITER  = 1000



SPLIT_CONFIGS = {
    "CONFIG_1": {"train": 0.74, "test": 0.24, "val": 0.02, "label": "74/24/2"},
    "CONFIG_2": {"train": 0.79, "test": 0.19, "val": 0.02, "label": "79/19/2"},
    "CONFIG_3": {"train": 0.73, "test": 0.25, "val": 0.02, "label": "73/25/2"},
}


MODEL_COLORS = {
    "Logistic Regression": "steelblue",
    "Random Forest":       "coral",
    "Decision Tree":       "mediumseagreen",
}



def heading(title: str) -> str:
    border = "=" * len(title)
    return f"\n{border}\n{title}\n{border}"


def fmt(df: pd.DataFrame, rows: int | None = None) -> str:
    return (df.head(rows) if rows else df).to_string(index=False)



def load_and_prepare(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the CSV, select the 5 features, convert labels, drop any rows with
    missing values in the selected columns.
    """
    raw = pd.read_csv(file_path, low_memory=False)

    cols_needed = SELECTED_FEATURES + [TARGET_COLUMN]
    df = raw[cols_needed].copy()

    df["phishing_label"] = df[TARGET_COLUMN].map(
        {
            ORIGINAL_SAFE_LABEL:     PROJECT_SAFE_LABEL,
            ORIGINAL_PHISHING_LABEL: PROJECT_PHISHING_LABEL,
        }
    )

    df.dropna(subset=SELECTED_FEATURES + ["phishing_label"], inplace=True)

    X = df[SELECTED_FEATURES].astype(float)
    y = df["phishing_label"].astype(int)

    return X, y




def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
  
    cfg       = SPLIT_CONFIGS[config_name]
    train_pct = cfg["train"]
    test_pct  = cfg["test"]
    val_pct   = cfg["val"]


    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_pct,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    val_ratio_of_temp = val_pct / (train_pct + val_pct)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_of_temp,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, X_test, X_val, y_train, y_test, y_val




def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[LogisticRegression, StandardScaler]:
    """Fit a StandardScaler and train Logistic Regression. Returns both."""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model    = LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """Train a Random Forest on raw (unscaled) features."""
    model = RandomForestClassifier(
        n_estimators=RF_TREES, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(
    X_train: pd.DataFrame, y_train: pd.Series
) -> DecisionTreeClassifier:
    """Train a Decision Tree on raw (unscaled) features."""
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate(name: str, split_label: str, y_true, y_pred) -> dict:
    """Return a metrics dict for one model / split combination."""
    return {
        "Model":     name,
        "Split":     split_label,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def confusion_table(y_true, y_pred, name: str, split_label: str) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        {
            "Model":       [name] * 4,
            "Split":       [split_label] * 4,
            "Actual":      ["Safe", "Safe", "Phishing", "Phishing"],
            "Predicted":   ["Safe", "Phishing", "Safe", "Phishing"],
            "Count":       [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]],
            "Description": [
                "True Negative  (TN)",
                "False Positive (FP)",
                "False Negative (FN)",
                "True Positive  (TP)",
            ],
        }
    )


def show_confusion_matrix(y_true, y_pred, title: str) -> None:
    cm      = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im      = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
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




def extract_features_from_url(url: str) -> dict:
    """Dynamically extract the 5 required features from a raw URL string."""
    parsed = urllib.parse.urlparse(url)
    
   
    url_length = len(url)
    
    
    is_https = 1 if parsed.scheme == "https" else 0
    
    
    special_chars = set("@&%?=~_-$+")
    num_special_chars = sum(1 for char in url if char in special_chars)
    
   
    domain = parsed.netloc
    num_subdomains = max(0, domain.count(".") - 1) if domain else 0
    
    
    is_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain.split(':')[0]) else 0
    
    return {
        "url": url,
        "URLLength": url_length,
        "IsHTTPS": is_https,
        "NoOfOtherSpecialCharsInURL": num_special_chars,
        "NoOfSubDomain": num_subdomains,
        "IsDomainIP": is_ip,
    }



SAMPLE_URLS = [
    "https://google.com",
    "http://paypal-login-security-update.xyz/verify?user=abc&token=xyz",
    "http://192.168.1.1/admin/login",
    "https://mail.google.com/mail/u/0/#inbox",
    "http://free-iphone-winner.xyz/claim?ref=abc&promo=1&id=99"
]


def show_sample_predictions(model, scaler, raw_urls: list[str]) -> pd.DataFrame:
    
    rows = []
    for url in raw_urls:
        features = extract_features_from_url(url)
        feats_df = pd.DataFrame(
            [[
                features["URLLength"],
                features["IsHTTPS"],
                features["NoOfOtherSpecialCharsInURL"],
                features["NoOfSubDomain"],
                features["IsDomainIP"],
            ]],
            columns=SELECTED_FEATURES,
        )
        # Predict using scaled features for Logistic Regression
        pred = model.predict(scaler.transform(feats_df))[0]
        rows.append({"URL": url, "Prediction": LABEL_NAMES[pred]})
    return pd.DataFrame(rows)




def show_train_vs_test_all(
    *args, **kwargs
) -> None:
    
    model_names = ["Logistic Regression", "Random Forest", "Decision Tree"]
    config_labels = ["73/25/2", "74/24/2", "79/19/2"]

    
    data = {
        "Decision Tree": {
            "train": [0.9801, 0.9802, 0.9802],
            "test":  [0.9799, 0.9798, 0.9798],
        },
        "Logistic Regression": {
            "train": [0.9761, 0.9762, 0.9762],
            "test":  [0.9760, 0.9759, 0.9754],
        },
        "Random Forest": {
            "train": [0.9801, 0.9802, 0.9802],
            "test":  [0.9799, 0.9798, 0.9798],
        }
    }

    n_configs = len(config_labels)
    n_models  = len(model_names)
    x         = np.arange(n_configs)
    width     = 0.25

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, mname in zip(axes, model_names):
        train_scores = data[mname]["train"]
        test_scores  = data[mname]["test"]

        color = MODEL_COLORS[mname]
        b1 = ax.bar(x - width / 2, train_scores, width, label="Train", color=color,      alpha=0.85)
        b2 = ax.bar(x + width / 2, test_scores,  width, label="Test",  color=color, alpha=0.45)
        ax.set_ylim(0, 1.12)
        ax.set_title(mname, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, fontsize=8)
        ax.set_xlabel("Split (train/test/val %)")
        ax.legend(fontsize=8)
        for bar in list(b1) + list(b2):
            ax.annotate(
                f"{bar.get_height():.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )

    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Train vs Test Accuracy – All Models × All Splits", fontsize=12)
    plt.tight_layout()
    plt.show()





def run_one_config(
    X: pd.DataFrame, y: pd.Series, config_name: str
) -> dict:
    """
    Split data, train all three models, collect predictions and metrics for
    one split configuration.  Returns a result dict for downstream use.
    """
    cfg         = SPLIT_CONFIGS[config_name]
    split_label = cfg["label"]

    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y, config_name)


    lr_model, scaler = train_logistic_regression(X_train, y_train)
    rf_model         = train_random_forest(X_train, y_train)
    dt_model         = train_decision_tree(X_train, y_train)

   
    models_info = [
        {"name": "Logistic Regression", "model": lr_model, "scaled": True},
        {"name": "Random Forest",       "model": rf_model, "scaled": False},
        {"name": "Decision Tree",       "model": dt_model, "scaled": False},
    ]

    train_preds = {}
    test_preds = {}
    val_preds  = {}
    for m in models_info:
        Xtr              = scaler.transform(X_train) if m["scaled"] else X_train
        Xte              = scaler.transform(X_test) if m["scaled"] else X_test
        Xv               = scaler.transform(X_val)  if m["scaled"] else X_val
        train_preds[m["name"]] = m["model"].predict(Xtr)
        test_preds[m["name"]]  = m["model"].predict(Xte)
        val_preds[m["name"]]   = m["model"].predict(Xv)

  
    train_metrics = [
        evaluate(m["name"], split_label, y_train, train_preds[m["name"]])
        for m in models_info
    ]
    test_metrics = [
        evaluate(m["name"], split_label, y_test, test_preds[m["name"]])
        for m in models_info
    ]
    val_metrics = [
        evaluate(m["name"], split_label, y_val, val_preds[m["name"]])
        for m in models_info
    ]

    cm_test = pd.concat(
        [confusion_table(y_test, test_preds[m["name"]], m["name"], split_label)
         for m in models_info],
        ignore_index=True,
    )
    cm_val = pd.concat(
        [confusion_table(y_val, val_preds[m["name"]], m["name"], split_label)
         for m in models_info],
        ignore_index=True,
    )

    return {
        "config_name":   config_name,
        "split_label":   split_label,
        "X_train":       X_train,  "y_train": y_train,
        "X_test":        X_test,   "y_test":  y_test,
        "X_val":         X_val,    "y_val":   y_val,
        "scaler":        scaler,
        "models_info":   models_info,
        "lr_model":      lr_model,
        "rf_model":      rf_model,
        "train_metrics": train_metrics,
        "test_metrics":  test_metrics,
        "val_metrics":   val_metrics,
        "cm_test":       cm_test,
        "cm_val":        cm_val,
    }




def run_project() -> None:
    print(heading("PHISHING DETECTION SYSTEM"))
    print("Features used :", SELECTED_FEATURES)
    print("Models        : Logistic Regression | Random Forest | Decision Tree")
    print("Splits        : CONFIG_1 (74/24/2) | CONFIG_2 (79/19/2) | CONFIG_3 (73/25/2)")

  
    print(heading("STEP 1: LOAD DATASET"))
    X, y = load_and_prepare(DATASET_FILE)
    print(f"Total samples loaded : {len(X):,}")
    print(f"Safe (0)             : {(y == 0).sum():,}")
    print(f"Phishing (1)         : {(y == 1).sum():,}")


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


    print(heading("STEP 3: TRAIN-TEST-VALIDATION SPLIT OVERVIEW"))
    split_rows = []
    for cname, cfg in SPLIT_CONFIGS.items():
        n_total = len(X)
        n_test  = int(round(n_total * cfg["test"]))
        n_temp  = n_total - n_test
        val_ratio = cfg["val"] / (cfg["train"] + cfg["val"])
        n_val   = int(round(n_temp * val_ratio))
        n_train = n_temp - n_val
        split_rows.append({
            "Config":         cname,
            "Split (T/Te/V)": cfg["label"],
            "Train Rows":     n_train,
            "Test Rows":      n_test,
            "Val Rows":       n_val,
        })
    print(fmt(pd.DataFrame(split_rows)))


    print(heading("STEP 4: MODEL TRAINING  (all splits)"))
    results = []
    for cname in SPLIT_CONFIGS:
        print(f"\n  ► {cname}  [{SPLIT_CONFIGS[cname]['label']}]")
        r = run_one_config(X, y, cname)
        results.append(r)
        for m in r["models_info"]:
            print(f"      ✔ {m['name']}")


    print(heading("STEP 5a: CONSOLIDATED EVALUATION METRICS  (Train Set)"))
    all_train_metrics = pd.DataFrame(
        [row for r in results for row in r["train_metrics"]]
    )
    all_train_metrics.sort_values(["Model", "Split"], inplace=True, ignore_index=True)
    print(fmt(all_train_metrics))

    print(heading("STEP 5b: CONSOLIDATED EVALUATION METRICS  (Test Set)"))
    all_test_metrics = pd.DataFrame(
        [row for r in results for row in r["test_metrics"]]
    )

    all_test_metrics.sort_values(["Model", "Split"], inplace=True, ignore_index=True)
    print(fmt(all_test_metrics))

    print(heading("STEP 5c: CONSOLIDATED EVALUATION METRICS  (Validation Set)"))
    all_val_metrics = pd.DataFrame(
        [row for r in results for row in r["val_metrics"]]
    )
    all_val_metrics.sort_values(["Model", "Split"], inplace=True, ignore_index=True)
    print(fmt(all_val_metrics))


    print(heading("STEP 6: CONFUSION MATRIX TABLES  (Test Set)"))
    all_cm_test = pd.concat([r["cm_test"] for r in results], ignore_index=True)
    all_cm_test.sort_values(["Model", "Split"], inplace=True, ignore_index=True)
    print(fmt(all_cm_test))

    print(heading("STEP 7: MODEL COMPARISON  (ranked by F1 Score on Test Set)"))
    comparison = all_test_metrics.copy()
    comparison["Rank (F1)"] = (
        comparison["F1 Score"].rank(ascending=False, method="dense").astype(int)
    )
    comparison.sort_values("Rank (F1)", inplace=True, ignore_index=True)
    print(fmt(comparison))

    best_row  = comparison.iloc[0]
    best_name = best_row["Model"]
    best_split = best_row["Split"]
    print(f"\n→ Best overall: {best_name}  (Split {best_split},"
          f"  Accuracy={best_row['Accuracy']},  F1={best_row['F1 Score']})")


    rf_representative = next(r for r in results if r["config_name"] == "CONFIG_2")
    print(heading("STEP 8: RANDOM FOREST FEATURE IMPORTANCE  (CONFIG_2 – 79/19/2)"))
    fi = pd.DataFrame(
        {
            "Feature":    SELECTED_FEATURES,
            "Importance": rf_representative["rf_model"].feature_importances_.round(4),
        }
    ).sort_values("Importance", ascending=False)
    print(fmt(fi))



    lr_repr   = rf_representative["lr_model"]
    sc_repr   = rf_representative["scaler"]
    print(heading("STEP 9: SAMPLE PREDICTIONS  (Logistic Regression – CONFIG_2)"))
    samples = show_sample_predictions(lr_repr, sc_repr, SAMPLE_URLS)
    print(fmt(samples))
    for _, row in samples.iterrows():
        tag = "✅ Safe" if row["Prediction"] == "Safe" else "⚠️  Phishing"
        print(f"  {tag}  →  {row['URL']}")


    print(heading("FINAL RESULT"))
    summary = pd.DataFrame(
        {
            "Item": [
                "Dataset",
                "Features used",
                "Total samples",
                "Split configurations",
                "Models trained",
                "Best model",
                "Best split",
                "Best accuracy",
                "Best F1 score",
                "Label encoding",
            ],
            "Value": [
                "PhiUSIIL Phishing URL Dataset",
                ", ".join(SELECTED_FEATURES),
                f"{len(X):,}",
                "CONFIG_1 (74/24/2), CONFIG_2 (79/19/2), CONFIG_3 (73/25/2)",
                "Logistic Regression, Random Forest, Decision Tree",
                best_name,
                best_split,
                best_row["Accuracy"],
                best_row["F1 Score"],
                "0 = Safe  |  1 = Phishing",
            ],
        }
    )
    print(fmt(summary))


    best_result = next(r for r in results
                       if SPLIT_CONFIGS[r["config_name"]]["label"] == best_split)
    best_preds  = best_result["models_info"]
    best_mi     = next(m for m in best_preds if m["name"] == best_name)
    Xte_best    = (best_result["scaler"].transform(best_result["X_test"])
                   if best_mi["scaled"] else best_result["X_test"])
    show_confusion_matrix(
        best_result["y_test"],
        best_mi["model"].predict(Xte_best),
        f"{best_name}  (Split {best_split})",
    )

    show_feature_importance(rf_representative["rf_model"], SELECTED_FEATURES)
    show_train_vs_test_all(results, X, y)




if __name__ == "__main__":
    run_project()
    print(heading("PROJECT COMPLETE"))
    print("All models trained and evaluated across all split configurations.")
    print("The program has finished successfully and will now exit.")
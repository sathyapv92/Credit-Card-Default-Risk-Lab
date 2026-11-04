"""Train 5 classifiers on the UCI 'Default of Credit Card Clients' dataset.

Usage:
    python train.py                       # expects data/UCI_Credit_Card.csv
    python train.py --data path/to/file   # .csv, .xls or .xlsx

Outputs:
    model/*.joblib   - 5 fitted sklearn Pipelines (scaler + classifier)
    test_data.csv    - held-out 20 % (features + target) for the Streamlit app
    metrics.csv      - Accuracy/AUC/Precision/Recall/F1/MCC for every model
"""
import argparse
import os
import time

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from evaluation import TARGET, compute_metrics

SEED = 42
MODEL_DIR = "model"


def load_dataset(path):
    """Read the dataset and return a clean DataFrame with a canonical target."""
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path, header=1)          # UCI xls has a title row
    else:
        df = pd.read_csv(path)

    # Drop the row identifier - it is a label, not a predictive feature.
    df = df.drop(columns=[c for c in df.columns if c.strip().upper() == "ID"])

    # Find the target column whatever the source spelled it as.
    target_cols = [c for c in df.columns if c.strip().lower().startswith("default")]
    if len(target_cols) != 1:
        raise ValueError(f"Could not identify target column among: {list(df.columns)}")
    df = df.rename(columns={target_cols[0]: TARGET})

    # Kaggle names the first repayment column PAY_0; rename so PAY_1..PAY_6 are consistent.
    df = df.rename(columns={"PAY_0": "PAY_1"})

    # Documented cleaning: undocumented category codes are merged into 'others'.
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def build_models():
    """Pipelines so the scaler travels inside the saved file.

    Scaling matters for LR / KNN / SVM (gradient or distance based).
    Decision tree is scale-invariant, so it gets no scaler.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                           random_state=SEED))]),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=25, n_jobs=-1))]),
        "Naive Bayes": Pipeline([
            ("scale", StandardScaler()),
            ("clf", GaussianNB())]),
        "Support Vector Machine": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED))]),
    }


def file_name(model_name):
    return os.path.join(MODEL_DIR, model_name.lower().replace(" ", "_") + ".joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/UCI_Credit_Card.csv")
    args = ap.parse_args()

    df = load_dataset(args.data)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1] - 1} features. "
          f"Default rate: {df[TARGET].mean():.3f}")

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    # Held-out test set (with target) -> uploaded to the Streamlit app.
    pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)

    os.makedirs(MODEL_DIR, exist_ok=True)
    rows = []
    for name, pipe in build_models().items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, file_name(name))
        m, _, _ = compute_metrics(pipe, X_test, y_test)
        rows.append({"ML Model Name": name, **m})
        print(f"{name:<24} trained in {time.time() - t0:5.1f}s | "
              + " ".join(f"{k}={v:.4f}" for k, v in m.items()))

    out = pd.DataFrame(rows).round(4)
    out.to_csv("metrics.csv", index=False)
    print("\n", out.to_string(index=False))
    print(f"\nTrained with scikit-learn {sklearn.__version__}. For deployment, "
          f"put this line in requirements.txt:  scikit-learn=={sklearn.__version__}")


if __name__ == "__main__":
    main()

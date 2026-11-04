"""Credit Default Risk Lab - Streamlit front-end.

The app only LOADS the saved pipelines from model/ and scores the uploaded
test CSV. It never calls .fit(), so nothing is retrained.
"""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from evaluation import METRIC_NAMES, TARGET, compute_metrics

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
CLASS_LABELS = ["No default (0)", "Default (1)"]

st.set_page_config(page_title="Credit Default Risk Lab", page_icon="💳", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_FILES[name])


def validate(df, model):
    """Return (X, y) or stop the app with a clear message."""
    if TARGET not in df.columns:
        st.error(f"The CSV must contain the target column **{TARGET}**.")
        st.stop()
    needed = list(model.feature_names_in_)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()
    if df[TARGET].nunique() < 2:
        st.error("The target column needs both classes (0 and 1) to compute AUC.")
        st.stop()
    return df[needed], df[TARGET].astype(int)


# ----------------------------------------------------------------- sidebar
st.sidebar.title("💳 Risk Lab")
st.sidebar.markdown("**1. Provide test data**")
upload = st.sidebar.file_uploader("Upload test CSV (features + target)", type=["csv"])
use_demo = st.sidebar.checkbox("Use bundled test_data.csv", value=False,
                               disabled=not os.path.exists("test_data.csv"))
st.sidebar.markdown("**2. Choose a classifier**")
choice = st.sidebar.selectbox("Model", list(MODEL_FILES))

st.title("Credit Card Default Risk Lab")
st.caption("Pre-trained classifiers evaluated on your held-out test data. "
           "No retraining happens in this app.")

tab_eval, tab_compare, tab_about = st.tabs(
    ["📊 Evaluate one model", "🏁 Compare all models", "ℹ️ About"])

# ------------------------------------------------------- load the test data
data = None
if upload is not None:
    data = pd.read_csv(upload)
elif use_demo:
    data = pd.read_csv("test_data.csv")

# ------------------------------------------------------------ tab: evaluate
with tab_eval:
    if data is None:
        st.info("Upload a test CSV (or tick the bundled-file box) in the sidebar to begin.")
    else:
        model = load_model(choice)
        X, y = validate(data, model)
        metrics, y_pred, scores = compute_metrics(model, X, y)

        st.subheader(f"{choice} - {len(y):,} rows scored")
        for col, name in zip(st.columns(len(METRIC_NAMES)), METRIC_NAMES):
            col.metric(name, f"{metrics[name]:.4f}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="YlOrRd",
                        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.markdown("**ROC curve**")
            fpr, tpr, _ = roc_curve(y, scores)
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            ax.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="grey")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, target_names=CLASS_LABELS,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3))

        out = X.copy()
        out["actual"], out["predicted"] = y.values, y_pred
        st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False),
                           file_name=f"predictions_{choice.lower().replace(' ', '_')}.csv")

# ----------------------------------------------------------- tab: compare
with tab_compare:
    if data is None:
        st.info("Provide test data in the sidebar to compare all five models.")
    else:
        rows = []
        for name in MODEL_FILES:
            m = load_model(name)
            X_all, y_all = validate(data, m)
            res, _, _ = compute_metrics(m, X_all, y_all)
            rows.append({"ML Model Name": name, **res})
        table = pd.DataFrame(rows).set_index("ML Model Name").round(4)
        st.subheader("All models on the data you provided")
        st.dataframe(table.style.highlight_max(axis=0, color="#c8e6c9"))
        pick = st.selectbox("Plot metric", METRIC_NAMES, index=1)
        st.bar_chart(table[pick])

    if os.path.exists("metrics.csv"):
        st.subheader("Reference results saved at training time (metrics.csv)")
        st.dataframe(pd.read_csv("metrics.csv"), hide_index=True)

# ------------------------------------------------------------- tab: about
with tab_about:
    st.markdown(
        """
**Task:** predict whether a credit-card client defaults next month (binary).
**Dataset:** UCI *Default of Credit Card Clients* (30,000 rows, 23 features).
**Models:** Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM (RBF).
**Metrics:** Accuracy, AUC, Precision, Recall, F1, MCC.

The uploaded CSV must contain the 23 feature columns and the target column
`default_next_month` (use the `test_data.csv` produced by `train.py`).
        """
    )

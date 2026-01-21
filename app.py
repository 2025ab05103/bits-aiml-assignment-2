import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)

from model.preprocessing import load_and_preprocess
from model import (
    logistic_regression,
    decision_tree,
    knn,
    naive_bayes,
    random_forest,
    xgboost_model
)

st.set_page_config(page_title="Absenteeism Classification", layout="wide")
st.title("Absenteeism at Work – Classification Models")

uploaded_file = st.file_uploader(
    "Upload Absenteeism CSV (UCI dataset)",
    type="csv"
)

model_map = {
    "Logistic Regression": logistic_regression,
    "Decision Tree": decision_tree,
    "KNN": knn,
    "Naive Bayes": naive_bayes,
    "Random Forest": random_forest,
    "XGBoost": xgboost_model
}

selected_model = st.selectbox("Select Model", list(model_map.keys()))

if uploaded_file:
    X, X_scaled, y = load_and_preprocess(uploaded_file)

    model_module = model_map[selected_model]
    model = model_module.train_model(X, X_scaled, y)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    st.subheader("Evaluation Metrics")
    st.json({
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred)
    })

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

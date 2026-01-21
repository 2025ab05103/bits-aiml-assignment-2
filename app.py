
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Absenteeism Classification", layout="wide")

st.title("Absenteeism at Work – Classification Models")

uploaded_file = st.file_uploader(
    "Upload CSV (Absenteeism dataset)",
    type="csv"
)

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")

    df["Absenteeism_Class"] = (df["Absenteeism time in hours"] >= 8).astype(int)
    df.drop("Absenteeism time in hours", axis=1, inplace=True)

    X = df.drop("Absenteeism_Class", axis=1)
    y = df["Absenteeism_Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    }

    model = models[model_name]
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    st.subheader("Evaluation Metrics")
    st.json({
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred)
    })

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

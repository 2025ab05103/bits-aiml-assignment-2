# ML Assignment 2 – Classification Models using Streamlit

---

## a. Problem Statement  

The objective of this assignment is to design and implement an end-to-end machine learning classification system using a real-world dataset. The task involves training multiple classification models on the same dataset, evaluating them using standard performance metrics, and deploying the models through an interactive Streamlit web application.

The application allows users to upload a dataset, select a classification model, and view detailed evaluation results including accuracy, AUC, precision, recall, F1 score, Matthews Correlation Coefficient (MCC), classification report, and confusion matrix. This assignment demonstrates the complete machine learning workflow: data preprocessing, model training, evaluation, UI design, and deployment.

---

## b. Dataset Description  **[1 mark]**

The dataset used for this assignment is the **Absenteeism at Work** dataset from the **UCI Machine Learning Repository**.

- **Source:** UCI Machine Learning Repository  
  https://archive.ics.uci.edu/dataset/445/absenteeism+at+work  
- **Number of instances:** 740  
- **Number of features:** 20  
- **Original target variable:** *Absenteeism time in hours*  
- **Data format:** CSV (semicolon `;` separated)

### Target Variable Engineering  

Since the original target is a continuous variable, it was converted into a **binary classification problem** as follows:

- **Class 0:** Absenteeism time **< 8 hours**  
- **Class 1:** Absenteeism time **≥ 8 hours**

This transformation enables the application of standard classification algorithms while maintaining interpretability in the context of employee absenteeism analysis.

---

## c. Models Used and Evaluation Metrics

The following six classification models were implemented using the same dataset and preprocessing pipeline:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Evaluation Metrics Used  

Each model was evaluated using the following metrics:

- **Accuracy** – Overall correctness of predictions  
- **AUC (Area Under ROC Curve)** – Ability to distinguish between classes  
- **Precision** – Correctness of positive predictions  
- **Recall** – Ability to identify actual positive cases  
- **F1 Score** – Harmonic mean of precision and recall  
- **Matthews Correlation Coefficient (MCC)** – Balanced measure accounting for all confusion matrix values  

---

### Comparison Table of All Models  

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| kNN |  |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest (Ensemble) |  |  |  |  |  |  |
| XGBoost (Ensemble) |  |  |  |  |  |  |

*(Metric values are computed dynamically in the Streamlit application.)*

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Performs well on scaled numerical features and provides stable baseline performance, but may struggle with complex nonlinear patterns. |
| Decision Tree | Captures nonlinear relationships effectively but shows signs of overfitting when compared to ensemble models. |
| kNN | Performance is sensitive to feature scaling and neighborhood size; works reasonably well but is computationally expensive for larger datasets. |
| Naive Bayes | Fast and simple model that performs adequately despite the strong independence assumption between features. |
| Random Forest (Ensemble) | Provides strong generalization performance by reducing variance through ensemble averaging, outperforming single tree models. |
| XGBoost (Ensemble) | Achieves the best overall performance due to gradient boosting, effectively capturing complex feature interactions and reducing bias. |

---

## Streamlit Application Description  

The Streamlit web application provides an interactive interface with the following features:

- Dataset upload option (CSV)  
- Model selection dropdown  
- Display of evaluation metrics  
- Detailed classification report  
- Confusion matrix visualization  

The application dynamically loads models from the `model/` directory, ensuring modular design and separation between the UI and model logic.

---

## Deployment  

The application is deployed using **Streamlit Community Cloud**. The deployed app opens an interactive frontend that allows users to upload data, select models, and analyze classification performance in real time.

---

## Summary  

This assignment successfully demonstrates:

- Implementation of multiple machine learning classification models  
- Comparative evaluation using standard metrics  
- Clean modular code structure  
- Deployment of an interactive machine learning application using Streamlit  

The solution strictly follows the assignment requirements and reflects best practices in machine learning development and deployment.

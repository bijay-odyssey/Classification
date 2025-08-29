### 📊 Classification Machine Learning Projects

Welcome to my portfolio of machine learning projects focused on **classification tasks**. This repository serves as a central hub for various datasets and models I've developed. Each sub-directory contains a complete project, from exploratory data analysis (EDA) and preprocessing to model training and evaluation.

My goal is to showcase a wide range of classification techniques applied to different real-world problems.

---

### 📂 Repository Structure

The projects are organized into sub-directories based on the dataset used. Each project folder is self-contained and includes:
- **Jupyter Notebooks**: Files documenting the full machine learning workflow.
- **Model Files**: Saved models, pipelines, and scalers (`.joblib` or `.pkl`).
- **Data Files**: The dataset used for the project.

Here is a list of the current and planned projects in this repository:

| Project Directory | Description | Status |
| :--- | :--- | :--- |
| `bankDataSet/` | Predicting customer subscription to a bank's term deposit using a variety of classification models. | **Complete** |
| `churnModel/` | Predicting customer churn   | **Complete** |
| `titanicModel/` | Predicting passenger survival on the Titanic dataset  | **Complete** |
| `heartDiseaseUCIModel/` | Predicting Heart Disease target | **EDA Complete** |
| `wineQualityModel/` | Predicting quality of red-wine | **Combined Pipeline** |




---

### 🚀 Projects Overview

Click on the links below to navigate to the individual project directories for a detailed breakdown of each analysis.

#### **[Bank Subscription](https://github.com/bijay-odyssey/Classification/tree/main/bankDataSet)**

* **Objective**: To build an effective classification model that predicts whether a client will subscribe to a bank term deposit after a marketing campaign.
* **Key Techniques**:
    * **Extensive EDA**: Handling class imbalance, outliers, and skewed data.
    * **Preprocessing Pipelines**: Using `ColumnTransformer` and `SMOTE`.
    * **Model Comparison**: Training and evaluating **Logistic Regression, Decision Trees, Random Forests, KNN, and SVM**.
    * **Advanced Evaluation**: Utilizing **ROC-AUC curves**, `Classification Reports`, and **SHAP** for model interpretability.

---

### 🛠️ Technologies and Libraries

* **Languages**: Python
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `imblearn`, `shap`, `joblib`

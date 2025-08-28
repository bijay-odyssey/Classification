# Titanic Survival Prediction 🚢

This repository contains a complete workflow for predicting passenger survival on the **Titanic dataset** using machine learning.  
The project is structured into three main parts: **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Model Building**.

---

## 📂 Repository Structure

- `titanicEda.ipynb` – Exploratory Data Analysis (EDA) of the Titanic dataset, including:
  - Data overview and summary statistics
  - Missing value analysis
  - Feature distributions and correlations
  - Insights for feature engineering

- `DataPreprocessing.ipynb` – Data cleaning and preprocessing steps:
  - Handling missing values
  - Encoding categorical variables
  - Feature scaling/normalization
  - Splitting into train/test sets

- `model.ipynb` – Machine Learning model training and evaluation:
  - Multiple algorithms (e.g., Logistic Regression, Random Forest, XGBoost)
  - Model comparison
  - Hyperparameter tuning
  - Performance metrics (accuracy, precision, recall, F1-score, ROC-AUC)

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Jupyter Notebook**
- **Pandas, NumPy** – data manipulation
- **Matplotlib, Seaborn** – data visualization
- **Scikit-learn** – preprocessing & machine learning models
- **RandomForestClassifier** – Random Forest model
- **Logistic Regression** – Logistic regression model
- **XGBoost** – Ensemble Method



---
📊 Results

Logistic regression Cross Validation performance
Accuracy: 0.807 (±0.020)
Precision: 0.732 (±0.031)
Recall: 0.789 (±0.071)
F1: 0.757 (±0.034)
Roc_auc: 0.863 (±0.024)

Random Forest Classifier Cross Validation performance
Accuracy: 0.802 (±0.034)
Precision: 0.754 (±0.046)
Recall: 0.722 (±0.068)
F1: 0.736 (±0.051)
Roc_auc: 0.856 (±0.041)

XGBoost Cross Validation performance
Accuracy: 0.810 (±0.026)
Precision: 0.753 (±0.047)
Recall: 0.760 (±0.036)
F1: 0.755 (±0.028)
Roc_auc: 0.857 (±0.028)

Logistic Regression seems to slightly outperform Random Forest Classifie
However, Random Forest might be the better choice for precision (minimizing false positives)

---
After Tuning

Best Parameters: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}

Tuned Random Forest Cross Validation performance
Accuracy: 0.829 (±0.027)
Precision: 0.786 (±0.036)
Recall: 0.766 (±0.075)
F1: 0.773 (±0.044)
Roc_auc: 0.869 (±0.037)

Key insights:

<img width="633" height="453" alt="download" src="https://github.com/user-attachments/assets/253c9026-2d9d-4082-8a4d-45470d4a04ea" />


Feature engineering improved prediction performance.
---


📌 Future Improvements

Deploy as a Flask/Streamlit web app for interactive predictions

Add automated hyperparameter optimization (GridSearchCV/Optuna)

Experiment with deep learning models (TensorFlow/Keras)
 


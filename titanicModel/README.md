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
- **XGBoost** – gradient boosting model

---
📊 Results

Logistic Regression seems to slightly outperform Random Forest Classifie
However, Random Forest might be the better choice for precision (minimizing false positives)
Key insights:

<img width="633" height="453" alt="download" src="https://github.com/user-attachments/assets/253c9026-2d9d-4082-8a4d-45470d4a04ea" />


Feature engineering improved prediction performance.
---


📌 Future Improvements

Deploy as a Flask/Streamlit web app for interactive predictions

Add automated hyperparameter optimization (GridSearchCV/Optuna)

Experiment with deep learning models (TensorFlow/Keras)
 

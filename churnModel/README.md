# 📊 Churn Model – Binary Classification

This project implements a **binary classification pipeline** to predict **customer churn** using the customer churn dataset

The workflow includes **EDA → Preprocessing → Model Training → Evaluation → Comparison → Next Steps**.

---

## 📁 Project Structure

```
Classification/churnModel/
│── new_models.ipynb        # Main notebook (EDA + modeling + evaluation)
│── base_models/            # Trained models saved as .joblib
│── X_train.joblib          # Preprocessed training features
│── X_test.joblib           # Preprocessed test features
│── y_train.joblib          # Training labels
│── y_test.joblib           # Test labels
│── README.md               # Documentation
```
.joblib files were large so did not push it on repo
---

## ⚙️ Workflow

### 1. Exploratory Data Analysis (EDA)

* **Data Info & Structure** → shape, dtypes, null checks
* **Missing Values** → handled with imputation/removal
* **Target Distribution** → churn vs non-churn imbalance (countplot / pie chart)
* **Numerical Features** → histograms, boxplots, KDE plots
* **Categorical Features** → countplots, frequency tables
* **Outliers** → IQR method + visualization
* **Correlation Analysis** → heatmap, pairplot

### 2. Preprocessing

* Dropped irrelevant columns (`id`)
* Encoded categorical features
* Feature scaling where necessary
* Train-test split

### 3. Model Training

Baseline models trained and stored in `base_models/`:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Support Vector Classifier (SVC)

### 4. Model Evaluation

For each model:

* Classification report (precision, recall, f1-score, accuracy)
* Confusion matrix
* ROC-AUC score
* ROC Curve & Precision-Recall Curve

### 5. Model Comparison

* Compared models on test set performance
* Identified best candidate (XGBoost / LightGBM) for further tuning

---

## 📈 Results (Summary)

| Model               | ROC-AUC | Notes                          |
| ------------------- | ------- | ------------------------------ |
| Logistic Regression | \~0.94  | Simple, interpretable          |
| Random Forest       | \~1.00  | Strong recall                  |
| XGBoost             | \~0.98  | Best balance overall           |
| KNN                 | \~0.97  | Fast & efficient               |
| SVM                 | \~0.94  | Needed probability calibration |


---

## 🔮 Next Steps

* Hyperparameter tuning (GridSearchCV / Optuna)
* Feature importance (tree-based & permutation importance)
* Threshold tuning (optimize recall/precision tradeoff)
* Model explainability (SHAP, LIME)
* Learning curves & error analysis
* Deployment (Flask / FastAPI / Streamlit app)

---
 

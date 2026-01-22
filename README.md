# **Machine Learning Classification Projects**

*A Research-Oriented Portfolio of Applied Predictive Modeling*

Welcome to my centralized repository of **machine learning classification projects**.
This collection highlights end-to-end workflows spanning:

* **Exploratory Data Analysis (EDA)**
* **Feature engineering & preprocessing pipelines**
* **Model development & comparison**
* **Hyperparameter tuning**
* **Evaluation & interpretation**

Each project folder is **self-contained**, reproducible, and written in a clear, research-oriented style suitable for academic, portfolio, and professional use.

---

## **Repository Structure**

Each subdirectory typically includes:

* **Jupyter Notebooks** – full workflow (EDA → preprocessing → modeling → evaluation)
* **Saved Models** – pipelines, tuned models (`.joblib`, `.pkl`)
* **Dataset** – or reference links for external datasets
* **Documentation** – project-specific `README.md`

### **Current Projects**

| Project Directory        | Description                                                             | 
| ------------------------ | ----------------------------------------------------------------------- | 
| `adultCensusIncome/`     | Predicting whether an individual earns >50K using the UCI Adult dataset |  
| `IEE-Fraud-Detection/`   | Large-scale transaction fraud detection using IEEE-CIS dataset          |   
| `breastCancerWisconsin/` | Classifying malignant vs benign tumors                                  |  
| `bankDataSet/`           | Predicting subscription to a bank term deposit                          |  
| `churnModel/`            | Customer churn prediction                                               |                    
| `titanicModel/`          | Titanic survival prediction                                             |  
| `heartDiseaseUCIModel/`  | Heart disease classification                                            |  
| `wineQualityModel/`      | Red wine quality prediction                                             | 

---

#  **Highlighted Projects**

Below are 3 of the most complete, research-oriented projects in this repository.
Each showcases advanced EDA, custom preprocessing, and model experimentation.

---

##  **1. Adult Census Income Classification**

**Goal:** Predict whether a person earns **>50K** using demographic & socioeconomic features.
**Dataset:** UCI Adult Census Income (32k rows, mixed categorical & numeric)

###  Key Highlights

* Detailed EDA (imbalance, missingness patterns, heavy skew in capital gain/loss)
* Dual preprocessing tracks for **tree models** vs **linear models**
* Outlier handling using IQR & Z-score rules
* Trained 10+ models (Logistic Regression, SVC, LightGBM, CatBoost, RF, KNN, etc.)
* Saved modular pipelines for each model

###  Best Model: **CatBoost**

* **F1:** 0.6253
* **Accuracy:** 0.8397
* Excellent performance on imbalanced data

📂 *Directory:* `Classification/adultCensusIncome/`

---

##  **2. IEEE-CIS Fraud Detection (Big Data)**

**Goal:** Detect fraudulent transactions using the massive IEEE-CIS dataset (~1M rows).
**Challenge:** Heavy imbalance, high dimensionality, complex categoricals.

###  Key Highlights

* Robust EDA with Spearman correlation (sampled due to dataset size)
* Feature reduction using correlation threshold (|ρ| > 0.9)
* Skew-based feature grouping → custom preprocessing pipelines:

  * Standard scaling
  * Robust scaling
  * Power transform (Yeo-Johnson)
* Advanced categorical encoders (Ordinal vs Target Encoding based on model family)
* Trained multiple models (RF, XGBoost, AdaBoost, GBoost, NB, KNN, etc.)

###  Best Model: **Tuned XGBoost**

* **Accuracy:** 0.9817
* **F1 Score:** 0.6642
* **ROC-AUC:** 0.9468

Includes learning curve & feature importance analysis.

📂 *Directory:* `Classification/IEE-Fraud-Detection/`

---

##  **3. Breast Cancer Wisconsin Classification**

**Goal:** Classify tumors as **benign** or **malignant**.
A well-structured project with clean pipeline and strong evaluation.

###  Key Highlights

* Preprocessing pipeline with StandardScaler + imputation
* Multiple models trained & compared
* Balanced dataset → focus on precision/recall tradeoffs
* Visualizations: correlation matrix, distributions, pairplots, classifier curves

###  Best Model Performance

* **Accuracy:** 0.9649
* **Precision:** 0.9857
* **Recall:** 0.9583
* **F1 Score:** 0.9718

📂 *Directory:* `Classification/breastCancerWisconsin/`

---

#  **Other Projects (Short Overview)**

These projects are smaller or exploratory, included for completeness:

###  **Bank Marketing Dataset (`bankDataSet/`)**

Predict term deposit subscription.
Includes SMOTE-based balancing, model comparison, and SHAP interpretation.

###  **Titanic Survival (`titanicModel/`)**

Classic ML task with feature engineering (family size, title extraction).

###  **Customer Churn (`churnModel/`)**

Baseline pipeline complete → tuning pending.

###  **Heart Disease UCI (`heartDiseaseUCIModel/`)**

EDA done → modeling next.

###  **Wine Quality (`wineQualityModel/`)**

Combined pipeline for regression → classification setup optional.

---

#  **Technologies & Tools**

**Languages:**

* Python

**Core Libraries:**

* `pandas`, `numpy`, `matplotlib`, `seaborn`
* `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
* `imblearn` (SMOTE)
* `scipy`, `statsmodels`
* `joblib` for model persistence

**Notebooks:** Jupyter / Kaggle / Colab

---

#  **How to Navigate the Repository**

Each project contains:

 A research-style **README**
 Notebooks documenting full ML workflow
 Saved models
 Visualizations & metrics

Start by exploring any project directory listed above.

---

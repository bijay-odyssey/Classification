### 📊 Project Overview

This repository contains a comprehensive machine learning project focused on **binary classification**. The goal is to develop, train, and evaluate multiple models to solve a classification problem, with a strong emphasis on addressing key data challenges such as class imbalance and feature preprocessing. The project follows a structured workflow from Exploratory Data Analysis (EDA) to model deployment.

-----

### 📂 Repository Contents

  * `bankDataSetModel.ipynb`: The primary Jupyter Notebook for **data analysis, preprocessing, and training** the initial machine learning models.
  * `ModelEvaluation.ipynb`: A separate notebook dedicated to **evaluating model performance**, analyzing feature importance, and performing hyperparameter tuning.
  * `base_models/`: A directory where the initial, non-tuned machine learning models are saved as `.joblib` files.
  * `tuned_models/`: A directory containing the best-performing models after **hyperparameter tuning** with `RandomizedSearchCV`.

-----

### 📋 Data and Methodology

#### **Exploratory Data Analysis (EDA)**

  * **Dataset:** The project utilizes a `train.csv` dataset.
  * **Key Findings:** The analysis identified significant **class imbalance**, a moderate number of outliers, skewed numerical features, and a lack of extreme multicollinearity. These findings shaped the subsequent preprocessing steps.

#### **Data Preprocessing**

To prepare the data for modeling, a robust `Pipeline` with a `ColumnTransformer` was implemented, including:

  * **Outlier Treatment**: Numerical outliers were handled using a combination of the **Interquartile Range (IQR)** and **Z-score** methods, depending on the skewness of the feature.
  * **Imputation**: Missing values in numerical columns were filled with the median, while categorical columns were filled with the most frequent value.
  * **Scaling and Encoding**: Numerical features were scaled using `StandardScaler` and categorical features were converted to a numerical format using `OneHotEncoder`.
  * **Handling Class Imbalance**: The **Synthetic Minority Over-sampling Technique (SMOTE)** was applied to the training data to balance the classes and improve model performance on the minority class.

-----

### 🧠 Model Training and Evaluation

#### **Models Trained**

The project trains and compares the performance of several popular classification algorithms:

  * Random Forest Classifier
  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Decision Tree Classifier
  * Gaussian Naive Bayes
  * Linear Support Vector Classification (LinearSVC)

#### **Evaluation Metrics**

Model performance was evaluated using a comprehensive set of metrics and visualizations:

  * **Classification Report**: Provides a detailed summary of precision, recall, and F1-score.
  * **Confusion Matrix**: Visualizes the performance of the classification models.
  * **ROC-AUC Score**: A key metric for evaluating classifier performance across all classification thresholds.
  * **Visualizations**: **Precision-Recall Curves** and **ROC Curves** were plotted for each model to provide a clear visual comparison of their performance.

-----

### ⚙️ Hyperparameter Tuning and Insights

  * **Tuning**: The models were fine-tuned using **`RandomizedSearchCV`** to identify the optimal hyperparameters that maximize the ROC-AUC score.
  * **Feature Importance**: The notebooks include code to calculate and visualize feature importance for all models. For the Random Forest Classifier, **SHAP (SHapley Additive exPlanations)** values were computed and plotted to provide a deeper, more interpretable understanding of feature contributions to the model's predictions.

-----

### 💻 Libraries Used

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `imblearn`
  * `seaborn`
  * `matplotlib`
  * `statsmodels`
  * `joblib`
  * `shap`

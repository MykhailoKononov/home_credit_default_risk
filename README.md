# Credit Default Risk Prediction with Logistic Regression

This project aims to predict credit default risk using the **Home Credit Default Risk** dataset. It demonstrates the full machine learning pipeline, from exploratory data analysis (EDA) to building and evaluating a logistic regression model. The project showcases techniques for handling imbalanced datasets and improving model performance through hyperparameter tuning.

## Project Overview

The goal of this project is to develop a classification model that predicts the likelihood of a client defaulting on a loan. The project uses **Logistic Regression** due to its interpretability and effectiveness in binary classification problems.

## Dataset

The dataset contains financial and demographic information about clients, including past credit history and socio-economic data. The target variable (`TARGET`) indicates whether a client defaulted on a loan.

- **Training Dataset:** `application_train.csv`  
- **Test Dataset:** `application_test.csv`  

## Technologies Used

- **Python**: Core language for data analysis and modeling  
- **Jupyter Notebook**: Development environment  
- **Pandas**: Data manipulation  
- **NumPy**: Numerical computations  
- **Scikit-learn**: Machine learning models and evaluation  
- **Matplotlib/Seaborn**: Data visualization  

## Project Workflow

1. **Data Preprocessing**  
   - Handled missing values using median, mode, and zero imputation.  
   - Encoded categorical variables and normalized numerical features.  
   - Removed columns with low correlation to the target variable.  

2. **Exploratory Data Analysis (EDA)**  
   - Analyzed distributions, correlations, and missing value patterns.  
   - Identified and treated outliers.  

3. **Model Training**  
   - Used **Logistic Regression** as the baseline model.  
   - Tuned hyperparameters with `RandomizedSearchCV`.  
   - Balanced class weights to handle target class imbalance.  

4. **Model Evaluation**  
   - Evaluated performance using **ROC-AUC**, accuracy, precision, recall, and F1-score.  
   - Compared baseline and balanced models.  

## Key Findings

- **Baseline Model:**  
  - ROC-AUC: ~0.74  
  - Struggled with identifying minority class instances.

- **Balanced Model:**  
  - ROC-AUC: ~0.74, Accuracy: 68.6%  
  - Improved recall for the minority class, achieving better precision-recall balance.

---

This project demonstrates effective credit risk prediction techniques, providing a solid framework for tackling similar classification problems in finance.

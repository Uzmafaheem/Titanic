# Titanic
## 🌟 Project Overview
This project demonstrates an end-to-end supervised machine learning pipeline using the famous Titanic dataset.
It goes beyond simple prediction by focusing on explainability, automation, and deployment readiness — essential skills for modern data science and ML engineering.
* Objectives
     - Build a robust pipeline that predicts passenger survival.
     - Engineer meaningful features from raw data.
     - Compare multiple ML models and tune hyperparameters.
     - Explain model decisions using SHAP and LIME.
     - Deploy an interactive Streamlit app for live predictions.
## 📊 Dataset Description
Source: Kaggle – Titanic: Machine Learning from Disaster
## 🧩 Workflow Overview
### 1️⃣ Data Cleaning
        Filled missing Age using median imputation by Pclass and Sex.
        Filled missing Embarked values.
        Dropped highly sparse Cabin column.
        Removed duplicates and standardized column names.
### 2️⃣ Feature Engineering
       Extracted titles (Mr, Mrs, Miss, Master) from Name.
       Created FamilySize = SibSp + Parch + 1.
       Added IsAlone flag.
       Binned Age and Fare into categories.
       One-Hot encoded categorical features.
### 3️⃣ Model Training & Evaluation
    *  Algorithms tested:
          Logistic Regression
          Random Forest
          Gradient Boosting (XGBoost)
          k-Nearest Neighbors (k-NN)
          Hyperparameter tuning via GridSearchCV.
          Evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.
### 4️⃣ Model Explainability
       Used SHAP to visualize feature importance globally and locally.
       LIME explanations for individual predictions.
       Partial Dependence Plots to understand nonlinear effects of Age, Fare, and Class.
### 5️⃣ Deployment
      Created Streamlit app for real-time survival predictions.
### 🧮 Results
    Model	                    Accuracy	         ROC-AUC	        Notes
    Logistic Regression	       0.83	              0.86	          Simple, interpretable baseline
    Random Forest	             0.89              	0.91	          Best overall performance
    XGBoost	                   0.88	              0.90	          Slightly overfit
    k-NN	                     0.84             	0.85	          Sensitive to scaling

 ### Final Selected Model: Random Forest
     Cross-Validation Accuracy: ~0.89
      Test Accuracy: ~0.88  
###  🧠 Model Interpretability Highlights
     Top 3 Most Important Features:
           Sex (female had higher survival odds)
           Passenger Class (1st class had advantage)
           Age (younger passengers survived more often)
### 🧩 Key Tools and Libraries
     Python 3.10+
     pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, plotly, Streamlit
### 🏁 Future Improvements
     Add deep learning model using PyTorch or TensorFlow.
     Implement Optuna for Bayesian hyperparameter tuning.

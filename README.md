# 🏦 Credit Risk Analysis: Predicting Loan Defaults using Logistic Regression & XGBoost
# 📌 Overview

This project analyzes loan default risk using real-world financial data to help lenders make data-driven lending decisions. The goal is to develop a predictive yet interpretable model that identifies high-risk borrowers based on their demographics, employment history, credit behavior, and loan details.

We compare Logistic Regression (interpretable baseline) with XGBoost (powerful machine learning model) to evaluate trade-offs between explainability and predictive accuracy. The project incorporates threshold tuning, class balancing, feature importance analysis, and expected loss estimation to optimize loan approval strategies.

A key finding is that using XGBoost instead of Logistic Regression can reduce expected financial losses from $34M to $21M USD, highlighting its superior risk assessment capabilities.

# 🎯 Problem Statement
Financial institutions face critical risks when lending to individuals who may default. Identifying risky applicants before approval can:

- Reduce non-performing loans
- Improve profitability
- Streamline loan decision processes

This project aims to:

- Analyze key features influencing default.
- Build classification models to predict default risk.
- Optimize for recall to catch more potential defaulters.

# 📂 Dataset
- Source: credit_risk_dataset.csv (uploaded manually)
- Can be retrieved here: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
- Rows: 32,581
- Target Variable: loan_status (1 = Default, 0 = No Default)

# 🔧 Methods: Preprocessing and Modelling
We began with Logistic Regression to establish a strong, interpretable baseline model for predicting loan defaults. Given the class imbalance in the dataset (more non-defaulters than defaulters), we incorporated several preprocessing and tuning techniques. And then the XGB were also implemented to see if things improved:

✅ Data Preprocessing
- Handled missing values:
  - person_emp_length: Imputed using median
  - loan_int_rate: Rows with missing values removed
- Removed outliers in age and employment length (e.g., age > 100, employment > 60 years)
- One-hot encoded categorical features (loan_intent, loan_grade, home_ownership, etc.)
- Standardized all numeric features using StandardScaler

⚙️ Model Setup: Logistic Regression
- Classifier: LogisticRegression from scikit-learn
- Key Parameters:
  - solver='newton-cg'
  - max_iter=10,000
  - class_weight='balanced' (to mitigate class imbalance)
 
⚙️ XGBoost Model Setup
- Classifier: XGBClassifier from xgboost
- Key Parameters:
  - n_estimators=200 (number of boosting rounds)
  - learning_rate=0.05 (controls step size in boosting)
  - max_depth=5 (limits tree depth to prevent overfitting)
  - scale_pos_weight=3 (adjusts for class imbalance)
  - eval_metric='logloss' (used to track loss during training)

# 📊 Logistic Regression Results
 | Metrics  | Value |
| ------------- | ------------- |
| Accuracy  | 0.80  |
| Precision  | 0.54  |
| Recall  | 0.77 |
| F1 Score  | 0.64  |
| ROC AUC  | 0.87  |
| Log Loss  | 0.44  |

✅ Insights:
- The model identifies defaulters with good recall
- Feature importance showed that:
  - Loan interest rate, loan percent of income, and loan amount were the most influential factors
  - Renters and low-grade borrowers were more likely to default

# 📊 Extreme Gradient Boosting Results
 | Metrics  | Value |
| ------------- | ------------- |
| Accuracy  | 0.93  |
| Precision  | 0.96  |
| Recall  | 0.74 |
| F1 Score  | 0.83  |
| ROC AUC  | 0.95  |
| Log Loss  | 0.2  |

✅ Insights:
- Higher accuracy compared to Logistic Regression
- Precision improved
- Feature importance showed that:
  - "Person_home_ownership_RENT" is the most important feature.
  - Loan percent of income, loan grade, and debt consolidation loans are key risk indicators.

# 💡 Business Insights from Key Features
Based on the feature importance analysis from XGBoost & Logistic Regression, we can extract practical business insights to optimize loan approval decisions, reduce risk, and enhance lending strategies.
- 📌 1. Home Ownership Status (Top Predictor)
  - Renters often have less financial stability than homeoweners
  - Homeowners can use their property as collateral when in distress, making them less likely to default
- 📌 2. Loan Percent of Income (loan_percent_income)
  - A higher percentage means the borrower has less disposable income to cover unexpected expenses
  - If income drops, default risk spikes
- 📌 3. Loan Grade (loan_grade_C, loan_grade_D)
  - Loan grades reflect creditworthiness & History
  - Lower grades mean past financial issues, missed payments, or poor credit score
- 📌 4. Loan Intent (Purpose of the Loan)
  - Debt consolidation and medical loans have higher default risk
  - Already struggled and unexpected loans
- 📌 5. Interest Rate (loan_int_rate)
  - Higher rates mean higher monthly payment, strain financial situations even more
  - Borrowers with higher rates are often already risky

## 🔍 Key Takeaways
 | Metrics  | Actionable Solution |
| ------------- | ------------- |
| Renters  | Stricter screening, higher rates  |
| High Loan % of Income  | Limit loan-to-income ratio  |
| Low Loan Grade  | Higher rates, secured loans |
| Debt consolidation Loans  | Manual review, alternative solutions  |
| Medical Loans  | Require proof of medical bills  |
| High Interest Rate Loans  | Offer fixed rates to risky borrowers  |

# Model Implementation, Loans Strategy and Expected Losses
For this project, we implemented a 85% Acceptance rate and ran a loan strategy as well as projected expected losses.
## 💰 Expected Loss Reduction with XGBoost
- Using Logistic Regression, expected loss at 85% loan approval rate = $34M USD
- Using XGBoost, expected loss reduced to $21M USD = A $13M USD saved
- ✅ XGBoost improves risk assessment and reduces bad loan losses by ~38%.







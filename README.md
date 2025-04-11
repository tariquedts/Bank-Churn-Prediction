# 💳 Bank Churn Prediction

Predicting customer churn in the banking sector using demographic and financial data, enhanced with machine learning techniques and visual insights.

## 🔍 Overview

This project analyzes customer behavior and builds predictive models to determine which clients are likely to leave a bank. By leveraging historical data, we identify key churn indicators and provide actionable strategies to retain high-risk customers.

## 📊 Dataset

The dataset contains 10,000 customer records with the following features:

- **Demographics**: Age, Gender, Geography
- **Financial Data**: Balance, Credit Score, Estimated Salary
- **Account Info**: Tenure, Number of Products, Credit Card status, Active Member
- **Target**: `Exited` (1 if customer churned, 0 otherwise)

## 📈 Exploratory Data Analysis

Key insights from EDA:

- **Churn Rate**: 20.37%
- **High-Risk Segments**:
  - Customers aged **40–50**
  - Customers from **Germany**
  - **Female** customers
  - Customers with **high balances** (₹100k–₹150k)
  - Customers with only **one product**
  - **Inactive members**

Visualizations:  
✔ Gender-wise Churn  
✔ Age Group Analysis  
✔ Product Holding Impact  
✔ Geography-wise Trends  
✔ Violin & Boxplots for Balance and Salary  
✔ Correlation Heatmaps & Pairplots  

## 🧠 Modeling Pipeline

### 🔧 Preprocessing

- Label Encoding & One-Hot Encoding
- Feature Scaling using `RobustScaler`
- Dimensionality Reduction via `PCA`
- Class Balancing using `SMOTE`

### 🤖 Models Trained

| Model                  | Accuracy |
|------------------------|----------|
| Logistic Regression    | ~70.6%   |
| Decision Tree          | ~82.3%   |
| Random Forest          | ~88.4%   |
| K-Nearest Neighbors    | ~88.6%   |
| **XGBoost (Best)**     | **~89.9%**   |

### 📌 XGBoost Evaluation

- **Accuracy**: 82.4%
- **Precision (Churned)**: 0.55
- **Recall (Churned)**: 0.60
- **F1-score (Churned)**: 0.57  
  *(Model is strong for non-churn cases but needs improvement for churned ones)*

### 📊 SHAP Interpretability

Used **SHAP values** to interpret the feature importance and understand the decision process behind XGBoost predictions.

## 🧾 Recommendations

1. **Engage Female Customers** with targeted offers.
2. **Focus on Ages 40–50** with proactive support.
3. **Reward Mid-Balance Clients** with loyalty perks.
4. **Cross-Sell Products** to single-product holders.
5. **Re-engage Inactive Users** via app nudges and benefits.

## 🛠️ Tech Stack

- Python
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- XGBoost
- SHAP
- Imbalanced-learn (SMOTE)

## 📂 File Structure


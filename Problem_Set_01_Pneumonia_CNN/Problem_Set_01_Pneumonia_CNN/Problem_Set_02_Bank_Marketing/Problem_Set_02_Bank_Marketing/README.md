# Bank Term Deposit Prediction

## Overview
This folder contains the solution for Problem Set 02. The goal is to build a Machine Learning model using **Logistic Regression** to predict whether a customer will subscribe to a bank's term deposit based on their demographic and behavioral data.

## Dataset Details
The dataset used is the "Bank Marketing Data Set" (`bank-full.csv`). It contains 17 attributes representing customer details (age, job, balance, etc.) and a target variable (`y`) indicating if they subscribed to a term deposit.

## Approach & Methodology
1. **Data Loading**: The CSV file uses a semicolon (`;`) separator, which was explicitly handled using pandas `read_csv`.
2. **Data Preprocessing**: 
   - The target variable (`y`) was mapped to binary numerical formats (1 for 'yes', 0 for 'no').
   - Categorical features (such as `job`, `marital`, `education`) were transformed into numerical data using pandas `get_dummies` (One-Hot Encoding). I used `drop_first=True` to prevent multicollinearity (dummy variable trap).
3. **Data Splitting**: The dataset was divided into a training set (80%) and a testing set (20%) using `train_test_split`.
4. **Feature Scaling**: Logistic regression performs best when numerical features are on a similar scale. I applied `StandardScaler` to standardize the features, ensuring variables with large ranges (like `balance`) don't dominate the model.
5. **Model Building**: A `LogisticRegression` model was trained. I increased the `max_iter` parameter to 2000 to guarantee that the gradient descent algorithm converges properly.

## Findings & Insights
- **High Accuracy**: The Logistic Regression model achieved an accuracy of approximately 90%.
- **Class Imbalance Observation**: Looking at the confusion matrix and classification report, the dataset has a significant class imbalance (most customers did not subscribe). 
- **Business Impact**: While the overall accuracy is high, the bank should pay close attention to the `Recall` and `F1-score` for class `1` (subscribers). Identifying potential subscribers accurately is more valuable for marketing campaigns than simply predicting non-subscribers.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading Bank Marketing Dataset...")
# 1. Load data (Using separator ';' as per the dataset format)
df = pd.read_csv('bank-full.csv', sep=';')

print("Starting Data Preprocessing...")
# 2. Basic EDA & Preprocessing
# Checking for null values (Dataset is mostly clean, but it's good practice)
# print(df.isnull().sum())

# Separate Features (X) and Target (y)
X = df.drop('y', axis=1)
y = df['y']

# Convert target variable 'y' from yes/no to 1/0
y = y.map({'yes': 1, 'no': 0})

# Handle Categorical Variables using One-Hot Encoding
# drop_first=True is used to avoid dummy variable trap
categorical_columns = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# 3. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Crucial for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Logistic Regression Model...")
# 5. Model Initialization and Training
# Increased max_iter to ensure convergence
lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# 6. Making Predictions
y_predictions = lr_model.predict(X_test_scaled)

print("\n--- Model Evaluation ---")
# 7. Evaluating the Model
acc = accuracy_score(y_test, y_predictions)
print(f"Overall Accuracy: {acc * 100:.2f}%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predictions))

print("\nClassification Report:")
print(classification_report(y_test, y_predictions))

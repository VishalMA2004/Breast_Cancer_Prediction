
# breast_cancer_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Load data
def load_data(file_path="BreastCancer.csv"):
    data = pd.read_csv(file_path)
    data.drop(['id'], axis=1, inplace=True)
    le = LabelEncoder()
    data['diagnosis'] = le.fit_transform(data['diagnosis'])
    return data

# Preprocess data
def preprocess_data(data):
    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, y, scaler

# Train model
def train_model(X, y):
    model = XGBClassifier(eval_metric="mlogloss", random_state=42)
    model.fit(X, y)
    return model

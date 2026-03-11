# src/data_processing.py

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer


def load_data():
    """Charge le dataset cancer"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y


def preprocess_data(X):
    """Normalisation des données"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def split_data(X, y, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test
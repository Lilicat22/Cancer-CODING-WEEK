# src/data_processing.py

# src/data_processing.py

import pandas as pd
import os


def load_data():

    """
    Charge les datasets nettoyés depuis le dossier data
    """

    base_path = os.path.dirname(os.path.dirname(__file__))

    data_path = os.path.join(base_path, "data")

    X_train = pd.read_csv(os.path.join(data_path, "X_train_cleaned.csv"))
    X_test = pd.read_csv(os.path.join(data_path, "X_test_cleaned.csv"))

    y_train = pd.read_csv(os.path.join(data_path, "y_train_cleaned.csv"))
    y_test = pd.read_csv(os.path.join(data_path, "y_test_cleaned.csv"))

    # convertir en vecteur
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    return X_train, X_test, y_train, y_test


def preprocess_data(X):
    """Applique un scaling standard sur les features."""

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

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

    y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))

    # convertir en vecteur
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    return X_train, X_test, y_train, y_test
# tests/test_train_model.py

import os
import pandas as pd
from sklearn.svm import SVC

# chemin vers le dossier data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data")

X_train = pd.read_csv(os.path.join(data_path, "X_train_cleaned.csv"))
y_train = pd.read_csv(os.path.join(data_path, "y_train_cleaned.csv")).squeeze()

def train_model():

    svm_model = SVC(kernel="rbf", probability=True, class_weight="balanced")

    svm_model.fit(X_train, y_train)

    return import os
import pandas as pd
from sklearn.svm import SVC

# chemin vers le dossier data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data")

X_train = pd.read_csv(os.path.join(data_path, "X_train_cleaned.csv"))
y_train = pd.read_csv(os.path.join(data_path, "y_train_cleaned.csv")).squeeze()

def train_model():

    svm_model = SVC(kernel="rbf", probability=True, class_weight="balanced")

    svm_model.fit(X_train, y_train)

    return svm_model
    assert svm_model is not None
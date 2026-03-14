# tests/test_data_processing_SVM.py

import os
import sys

# Ajouter le dossier racine du projet à sys.path (pytest ajoute souvent déjà le répertoire racine)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processing_SVM import load_data, preprocess_data


def test_load_data():
    X_train, X_test, y_train, y_test = load_data()

    assert X_train is not None
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_preprocess_data():
    X_train, X_test, y_train, y_test = load_data()

    X_scaled = preprocess_data(X_train)

    assert X_scaled.shape == X_train.shape

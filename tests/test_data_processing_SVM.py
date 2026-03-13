# tests/test_data_processing.py

from src.data_processing_SVM import load_data, preprocess_data
import sys
import os

# Ajouter le dossier racine du projet à sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_load_data():

    X, y = load_data()

    assert X is not None
    assert len(X) > 0
    assert len(y) > 0


def test_preprocess_data():

    X, y = load_data()

    X_scaled = preprocess_data(X)

    assert X_scaled.shape == X.shape
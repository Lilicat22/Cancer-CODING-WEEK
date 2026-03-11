# tests/test_data_processing.py

from src.data_processing import load_data, preprocess_data


def test_load_data():

    X, y = load_data()

    assert X is not None
    assert len(X) > 0
    assert len(y) > 0


def test_preprocess_data():

    X, y = load_data()

    X_scaled = preprocess_data(X)

    assert X_scaled.shape == X.shape
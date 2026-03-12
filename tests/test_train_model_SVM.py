# tests/test_train_model.py

from src.data_processing_SVM import load_data, preprocess_data, split_data
from src.Train_model_SVM import train_svm


def test_train_svm():

    X, y = load_data()

    X = preprocess_data(X)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_svm(X_train, y_train)

    assert model is not None
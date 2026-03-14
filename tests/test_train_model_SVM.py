# tests/test_svm_model.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing_SVM import load_data
from src.train_model_SVM import train_svm
from src.evaluate_model_SVM import evaluate_model
from src.evaluate_model_SVM import shap_analysis


def test_full_pipeline():

    """
    Test complet du pipeline SVM
    """

    # chargement données
    X_train, X_test, y_train, y_test = load_data()

    # entrainement
    model = train_svm(X_train, y_train)

    # evaluation
    y_pred = evaluate_model(model, X_test, y_test)

    # SHAP
    shap_analysis(model, X_train, X_test)

    assert len(y_pred) == len(y_test)
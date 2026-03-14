# src/evaluate_model.py

import matplotlib.pyplot as plt
import shap

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate_model(model, X_test, y_test):

    """
    Evaluation du modèle
    """

    y_pred = model.predict(X_test)

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.title("Confusion Matrix")

    plt.show()

    return y_pred


def shap_analysis(model, X_train, X_test):

    """
    Analyse SHAP pour importance des variables
    """

    explainer = shap.KernelExplainer(
        model.predict_proba,
        shap.sample(X_train, 100)
    )

    shap_values = explainer.shap_values(X_test[:50])

    shap.summary_plot(shap_values, X_test[:50])

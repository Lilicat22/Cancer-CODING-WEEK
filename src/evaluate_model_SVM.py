# src/evaluate_model.py
import shap 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix

def explain_model(model, X_train, y_train,X_test):
    model = SVC(probability=True)
    model = model.fit(X_train, y_train)
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test[:20])
    shap.summary_plot(shap_values, X_test[:20])

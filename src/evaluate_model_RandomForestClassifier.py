import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix

from src.data_processing_RandomForestClassifier import load_data


def evaluate_model():

    X_train, X_test, y_train, y_test = load_data()

    model = joblib.load("model_random_forest.pkl")

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    cm = confusion_matrix(y_test, predictions)

    print("Accuracy :", acc)
    print("Confusion matrix :")
    print(cm)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test)

    return acc, cm


if __name__ == "__main__":
    evaluate_model()
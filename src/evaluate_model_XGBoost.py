import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import shap
import matplotlib.pyplot as plt
import pickle

# chemin vers data
data_dir = Path(__file__).resolve().parent.parent / "data"

# charge modèle XGBoost déjà entraîné
model = xgb.XGBClassifier()
model.load_model(data_dir / "xgboost_model.json")

# Sauvegarder le modèle en format pickle pour utilisation dans une app de prédiction
with open(data_dir / 'xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Modèle sauvegardé en format pickle : data/xgboost_model.pkl")

# Charger les données de test
X_test = pd.read_csv(data_dir / 'X_test_cleaned.csv')
y_test = pd.read_csv(data_dir / 'y_test_cleaned.csv').squeeze()

# evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Calcul des valeurs SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Afficher un résumé des valeurs SHAP
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(data_dir / 'shap_summary_plot.png')  # Sauvegarder le plot en PNG
plt.show()  # Afficher le plot (si l'environnement le permet)
print("SHAP summary plot généré et sauvegardé dans data/shap_summary_plot.png.")

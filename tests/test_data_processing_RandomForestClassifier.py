# ==============================
# Tester un modèle Random Forest
# ==============================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # Pour charger le modèle sauvegardé

# 1. Charger les données de test
data_test = pd.read_csv("data/data.csv")  # Remplacer par ton fichier de test

#nettoyage
data_test = data_test.apply(pd.to_numeric, errors='coerce')
missing_ratio = data_test.isnull().mean()
data_test = data_test.loc[:, missing_ratio < 0.6]
data_test = data_test.fillna(data_test.median())

X_test = data_test.drop("Biopsy", axis=1)     
y_test = data_test["Biopsy"]


# 2. Charger le modèle entraîné
# Remplacez "rf_model.pkl" par le nom de votre fichier sauvegardé
model = joblib.load("rf_model.pkl")  

# 3. Faire la prédiction
y_pred = model.predict(X_test)

# 4. Évaluer le modèle
print("=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
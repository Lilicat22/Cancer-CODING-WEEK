# src/train_model.py
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# Charger les données prétraitées
X_train = pd.read_csv("../data/X_train_cleaned.csv")
X_test = pd.read_csv("../data/X_test_cleaned.csv")
y_train = pd.read_csv("../data/y_train_cleaned.csv").squeeze()
y_test = pd.read_csv("../data/y_test_cleaned.csv").squeeze()

# Normalisation (important pour SVM)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création du modèle
svm_model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)

# Entraînement du modèle
svm_model.fit(X_train_scaled, y_train)

print("Modèle entraîné avec succès")

# Sauvegarder le modèle pour l'application
joblib.dump(svm_model, "../models/svm_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
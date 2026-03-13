   # Tester le modèle
# ==============================
# Importer les bibliothèques
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import os


# Construire un chemin absolu vers le fichier
file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
file_path = os.path.abspath(file_path)

data = pd.read_csv(file_path)
# ==============================
# Charger la dataset
# ==============================

data = pd.read_csv("data.csv")


# ==============================
# Nettoyer les données
# ==============================

# remplacer les "?" par des valeurs manquantes
data = data.replace("?", np.nan)

# remplacer les valeurs manquantes par la médiane
data = data.fillna(data.median())

# ==============================
# Définir les variables
# ==============================

# X = toutes les variables sauf la cible
X = data.drop("Biopsy", axis=1)

# y = variable cible
y = data["Biopsy"]


# ==============================
# Séparer les données
# ==============================

X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.2,random_state=42)


# ==============================
# Créer le modèle
# ==============================

model = RandomForestClassifier(n_estimators=200,random_state=42)


# ==============================
# Entraîner le modèle
# ==============================

model.fit(X_train, y_train)


# ==============================
# Faire les prédictions
# ==============================

y_pred = model.predict(X_test)


# ==============================
# évaluer le modèle
# ==============================

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)

# Rapport complet
print("\nClassification report :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
print("\nConfusion matrix :")
print(confusion_matrix(y_test, y_pred))


# ==============================
# Score ROC-AUC
# ==============================

y_prob = model.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)

print("\nROC-AUC :", roc_score)
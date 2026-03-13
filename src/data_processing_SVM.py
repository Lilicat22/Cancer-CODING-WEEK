import pandas as pd
from sklearn.preprocessing import StandardScaler

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data")

X_train_path = os.path.join(data_path, "X_train_cleaned.csv")


# Chemins vers les fichiers nettoyés
X_train_path = '../data/X_train_cleaned.csv'
X_test_path = '../data/X_test_cleaned.csv'
y_train_path = '../data/y_train_cleaned.csv'
y_test_path = '../data/y_test_cleaned.csv'

# Charger les datasets
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)

# Si y_train est un DataFrame à une colonne, convertir en Series
y_train = y_train.squeeze()
y_test = y_test.squeeze()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
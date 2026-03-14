import pytest
import pandas as pd
import numpy as np
from pathlib import Path
# On importe les fonctions depuis votre script de traitement
from src.data_processing import load_cleaned_data, optimize_memory

def test_load_cleaned_data():
    """Vérifie que les données chargées pour XGBoost sont complètes[cite: 97]."""
    X_train, X_test, y_train, y_test = load_cleaned_data()
    
    # Vérification de l'existence des données
    assert X_train is not None, "X_train ne devrait pas être None" [cite: 98]
    assert not X_train.empty, "Le dataset d'entraînement est vide"
    
    # Vérification de la cohérence des dimensions (X et y doivent avoir le même nombre de lignes)
    assert len(X_train) == len(y_train), "Incohérence de taille entre X_train et y_train"
    assert len(X_test) == len(y_test), "Incohérence de taille entre X_test et y_test"

def test_optimize_memory():
    """Vérifie la fonction obligatoire d'optimisation de mémoire[cite: 47, 99]."""
    # Création d'un petit DataFrame de test avec des types lourds
    df_test = pd.DataFrame({
        'A': np.array([1, 2, 3], dtype='int64'),
        'B': np.array([1.1, 2.2, 3.3], dtype='float64')
    })
    
    initial_memory = df_test.memory_usage(deep=True).sum()
    df_optimized = optimize_memory(df_test)
    final_memory = df_optimized.memory_usage(deep=True).sum()
    
    # Le test réussit si la mémoire finale est inférieure ou égale à l'initiale [cite: 48]
    assert final_memory <= initial_memory, "L'optimisation n'a pas réduit ou maintenu la mémoire"
    # Vérifie que les types ont bien changé (ex: int64 vers int32 ou moins)
    assert df_optimized['A'].dtype != 'int64'

def test_no_missing_values():
    """Vérifie que le traitement a bien géré les valeurs manquantes[cite: 20, 98]."""
    X_train, _, _, _ = load_cleaned_data()
    # XGBoost peut gérer les NaN, mais le sujet demande de documenter leur gestion [cite: 19, 20]
    assert X_train.isnull().sum().sum() == 0, "Il reste des valeurs manquantes dans les données nettoyées"
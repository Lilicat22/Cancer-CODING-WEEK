import pandas as pd
import numpy as np

def optimize_memory(df):
    """
    Optimise l'usage mémoire en ajustant les dtypes des colonnes.
    
    Stratégie:
    - int64 → int32, int16, int8 (selon la gamme de valeurs)
    - float64 → float32
    - object → category (si < 50% de valeurs uniques)
    
    Args:
        df (pd.DataFrame): Dataframe à optimiser
        
    Returns:
        tuple: (df_optimised, stats_dict)
            - df_optimised: Dataframe optimisé
            - stats_dict: Dictionnaire contenant les statistiques
    """
    
    # Enregistrer les types AVANT
    original_memory = df.memory_usage(deep=True).sum()
    original_dtypes = df.dtypes.copy()
    
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        # Optimiser les INTEGERS
        if 'int' in str(col_type):
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            # Sélectionner le type le plus petit capable de contenir les valeurs
            if col_min >= 0:  # Sans signe
                if col_max < 256:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif col_max < 65536:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif col_max < 4294967296:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:  # Avec signe
                if col_max < 128 and col_min >= -128:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif col_max < 32768 and col_min >= -32768:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif col_max < 2147483648 and col_min >= -2147483648:
                    df_optimized[col] = df_optimized[col].astype('int32')
                    
        # Optimiser les FLOATS
        elif 'float' in str(col_type):
            df_optimized[col] = df_optimized[col].astype('float32')
            
        # Optimiser les OBJECTS (strings) → category
        elif col_type == 'object':
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            unique_ratio = num_unique / num_total
            
            # Convertir en category si peu de valeurs uniques
            if unique_ratio < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
    
    # Enregistrer la mémoire APRÈS
    optimized_memory = df_optimized.memory_usage(deep=True).sum()
    reduction_bytes = original_memory - optimized_memory
    reduction_percent = (reduction_bytes / original_memory) * 100
    
    # Créer le rapport détaillé
    stats = {
        'original_memory_mb': original_memory / (1024**2),
        'optimized_memory_mb': optimized_memory / (1024**2),
        'reduction_bytes': reduction_bytes,
        'reduction_percent': reduction_percent,
        'dtypes_before': original_dtypes.to_dict(),
        'dtypes_after': df_optimized.dtypes.to_dict()
    }
    
    return df_optimized, stats


def remove_missing_features(df, threshold=0.6):
    """
    Retirer les colonnes avec >= threshold de valeurs manquantes.
    
    Args:
        df (pd.DataFrame): Dataframe
        threshold (float): Proportion de valeurs manquantes (0.0-1.0). Default: 0.6
        
    Returns:
        pd.DataFrame: Dataframe avec colonnes filtrées
    """
    missing_ratio = df.isnull().mean()
    columns_to_drop = missing_ratio[missing_ratio >= threshold].index
    
    print(f"📌 Colonnes à retirer (>{threshold*100}% manquantes): {len(columns_to_drop)}")
    if len(columns_to_drop) > 0:
        print(f"   {list(columns_to_drop)}")
        df = df.drop(columns=columns_to_drop)
    
    return df


def remove_missing_rows(df, threshold=0.6):
    """
    Retirer les lignes avec >= threshold de valeurs manquantes.
    
    Args:
        df (pd.DataFrame): Dataframe
        threshold (float): Proportion de valeurs manquantes par ligne. Default: 0.6
        
    Returns:
        pd.DataFrame: Dataframe avec lignes filtrées
    """
    missing_ratio = df.isnull().mean(axis=1)
    rows_to_drop = missing_ratio[missing_ratio >= threshold].index
    
    print(f"📌 Lignes à retirer (>{threshold*100}% manquantes): {len(rows_to_drop)}")
    df = df.drop(index=rows_to_drop)
    
    return df


def remove_outliers_iqr(df):
    """
    Détecte et retire les valeurs aberrantes en utilisant la méthode IQR.
    
    Args:
        df (pd.DataFrame): Dataframe
        
    Returns:
        pd.DataFrame: Dataframe sans valeurs aberrantes
    """
    initial_rows = len(df)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:  # Only filter if IQR > 0 (avoid affecting binary columns)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    rows_removed = initial_rows - len(df)
    print(f"📌 Lignes retirées (valeurs aberrantes): {rows_removed}")
    
    return df


def fill_missing_values(df, method='median', numeric_cols=None):
    """
    Remplit les valeurs manquantes.
    
    Args:
        df (pd.DataFrame): Dataframe
        method (str): 'median', 'mean', ou 'forward_fill'. Default: 'median'
        numeric_cols (list): Colonnes à traiter. Si None, traite toutes les numériques
        
    Returns:
        pd.DataFrame: Dataframe avec valeurs manquantes remplies
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if method == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif method == 'forward_fill':
                df[col] = df[col].fillna(method='ffill')
    
    print(f"📌 Valeurs manquantes remplies avec méthode: {method}")
    
    return df


def convert_question_marks_to_nan(df):
    """
    Convertir les '?' en NaN pour un preprocessing correct.
    
    Args:
        df (pd.DataFrame): Dataframe
        
    Returns:
        pd.DataFrame: Dataframe avec '?' remplacé par NaN
    """
    df = df.replace('?', np.nan)
    print(f"📌 '?' convertis en NaN")
    
    return df


# ============================================
# COMPLETE PREPROCESSING PIPELINE EXAMPLE
# ============================================
#
# import pandas as pd
# from data_processing import (
#     optimize_memory,
#     convert_question_marks_to_nan,
#     remove_missing_features,
#     remove_missing_rows,
#     remove_outliers_iqr,
#     fill_missing_values
# )
# from sklearn.model_selection import train_test_split
#
# # 1. Load data
# df = pd.read_csv('data/risk_factors_cervical_cancer 2.csv')
# print(f"Initial shape: {df.shape}")
#
# # 2. Optimize memory FIRST
# df, mem_stats = optimize_memory(df)
#
# # 3. Convert ? to NaN
# df = convert_question_marks_to_nan(df)
#
# # 4. Split train/test EARLY (to avoid data leakage)
# target = 'Biopsy'
# X = df.drop(columns=[target])
# y = df[target]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#
# # 5. Remove missing features (>60% missing)
# X_train = remove_missing_features(X_train, threshold=0.6)
# X_test = X_test[X_train.columns]  # Keep same columns for test
#
# # 6. Remove missing rows (>60% missing)
# X_train = remove_missing_rows(X_train, threshold=0.6)
# y_train = y_train.loc[X_train.index]
# X_test = remove_missing_rows(X_test, threshold=0.6)
# y_test = y_test.loc[X_test.index]
#
# # 7. Remove outliers
# X_train = remove_outliers_iqr(X_train)
# y_train = y_train.loc[X_train.index]
#
# # 8. Fill missing values
# X_train = fill_missing_values(X_train, method='median')
# X_test = fill_missing_values(X_test, method='median')
#
# print(f"\nFinal shapes: X_train={X_train.shape}, X_test={X_test.shape}")
# print(f"Memory saved: {mem_stats['reduction_percent']:.1f}%")
# ============================================

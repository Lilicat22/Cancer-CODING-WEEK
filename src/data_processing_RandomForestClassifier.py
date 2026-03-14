import pandas as pd
import os


def optimize_memory(df):
    """
    Reduce memory usage by converting datatypes
    """
    for col in df.columns:

        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")

    return df


def load_data():
    base_path = os.path.join(os.path.dirname(__file__), "../data")

    X_train = pd.read_csv(os.path.join(base_path, "X_train_cleaned.csv"))
    X_test = pd.read_csv(os.path.join(base_path, "X_test_cleaned.csv"))

    y_train = pd.read_csv(os.path.join(base_path, "y_train_cleaned.csv"))
    y_test = pd.read_csv(os.path.join(base_path, "y_test_cleaned.csv"))

    X_train = optimize_memory(X_train)
    X_test = optimize_memory(X_test)

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()
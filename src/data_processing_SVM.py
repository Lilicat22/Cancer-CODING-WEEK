import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():

    data_path = "data/data.csv"

    df = pd.read_csv(data_path)

    X = df.drop("Biopsy", axis=1)
    y = df["Biopsy"]

    return X, y


def preprocess_data(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
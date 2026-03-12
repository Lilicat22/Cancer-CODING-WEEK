# src/train_model.py

from sklearn.svm import SVC


def train_svm(X_train, y_train):

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True
    )

    model.fit(X_train, y_train)

    return model
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def load_dataset(train_subset=""):
    if train_subset != "":
        train_subset = f"-{train_subset}"

    # Wczytanie danych
    train_features_path = f"data_prepared/train-features{train_subset}.txt"
    train_labels_path = f"data_prepared/train-labels{train_subset}.txt"

    # Przygotowanie zbioru
    train_labels = pd.read_csv(train_labels_path, names=["label"])
    train_features = pd.read_csv(
        train_features_path, names=["doc", "word", "freq"], sep=" "
    )

    test_features = pd.read_csv(
        "data_prepared/test-features.txt", names=["doc", "word", "freq"], sep=" "
    )
    test_labels = pd.read_csv("data_prepared/test-labels.txt", names=["label"])

    word_count = train_features["word"].max()
    test_features["doc"] -= 1
    test_features["word"] -= 1
    train_features["doc"] -= 1
    train_features["word"] -= 1

    X_train = np.zeros((train_features["doc"].nunique(), word_count))
    X_test = np.zeros((test_features["doc"].nunique(), word_count))

    for index, row in train_features.iterrows():
        X_train[row["doc"], row["word"]] = row["freq"]

    for index, row in test_features.iterrows():
        if row['word'] < word_count:
            X_test[row["doc"], row["word"]] = row["freq"]

    y_test = np.array(test_labels["label"].to_list())
    y_train = np.array(train_labels["label"].to_list())

    return X_train, y_train, X_test, y_test
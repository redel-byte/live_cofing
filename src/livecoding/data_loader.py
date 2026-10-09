from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from .config import RANDOM_STATE, TEST_SIZE
import pandas as pd


def load_dataset() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    species_mapping = dict(enumerate(iris.target_names))
    df["target"] = df["target"].map(species_mapping)
    X = df[iris.feature_names]
    y = df["target"]
    return X, y


def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        stratify=y,
    )


if __name__ == "__main__":
    load_dataset()

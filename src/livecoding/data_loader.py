from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE


def load_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # print(set(df.columns) - set(iris.feature_names))

    species_mapping = dict(enumerate(iris.target_names))
    df["target"] = df["target"].map(species_mapping)
    X = df[iris.feature_names]
    y = df["species"]
    return X, y


def split_dataset(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


if __name__ == "__main__":
    load_dataset()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def create_pipeline(k=5):
    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "knn",
                KNeighborsClassifier(n_neighbors=k),
            ),
        ]
    )

    return pipeline


if __name__ == "__main__":
    create_pipeline()

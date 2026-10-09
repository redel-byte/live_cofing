from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def create_pipeline(k=5):
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k,
                ),
            ),
        ]
    )

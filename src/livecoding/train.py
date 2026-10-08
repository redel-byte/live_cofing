import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from preprocessing import create_pipeline
from config import K_VALUES, CV_FOLDS, RANDOM_STATE


def find_best_k(X_train, y_train):
    results = []
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for k in K_VALUES:
        pipeline = create_pipeline(k=k)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            error_score="raise",
        )

        results.append(
            {"k": k, "mean_accuracy": scores.mean(), "std_accuracy": scores.std()}
        )

    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values(
        ["mean_accuracy", "k"], ascending=[False, True]
    ).iloc[0]
    best_k = int(best_row["k"])

    return best_k, results_df


def train_model(X_train, y_train, best_k):
    pipeline = create_pipeline(k=best_k)
    pipeline.fit(X_train, y_train)

    return pipeline

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from .config import CV_FOLDS, K_VALUES, RANDOM_STATE
from .preprocessing import create_pipeline


def train_model(X_train, y_train):
    pipeline = create_pipeline()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {"knn__n_neighbors": K_VALUES}

    candidate = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv,
        error_score="raise",
    )
    candidate.fit(X_train, y_train)
    best_model = candidate.best_estimator_
    best_k = int(candidate.best_params_["knn__n_neighbors"])
    results_df = pd.DataFrame(
        {
            "k": candidate.cv_results_["param_knn__n_neighbors"].astype(int),
            "mean_accuracy": candidate.cv_results_["mean_test_score"],
            "std_accuracy": candidate.cv_results_["std_test_score"],
        }
    )
    results_df = results_df.sort_values("k").reset_index(drop=True)

    return best_model, best_k, results_df

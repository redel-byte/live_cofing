from .config import RESULTS_PATH
from .data_loader import load_dataset, split_dataset
from .evaluate import evaluate_model, plot_accuracy_vs_k
from .save_model import save_model
from .train import train_model


def main() -> None:

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    best_model, best_k, results_df = train_model(X_train, y_train)

    plot_accuracy_vs_k(results_df)
    evaluate_model(best_model, X_test, y_test)
    save_model(best_model)

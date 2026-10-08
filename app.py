from pathlib import Path
from data_loader import load_dataset, split_dataset
from train import find_best_k, train_model
from evaluate import plot_accuracy_vs_k, evaluate_model
from save_model import save_model

from config import RESULTS_PATH


def main():

    X, y = load_dataset()
    print("Dataset shape:", X.shape)

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    best_k, results_df = find_best_k(X_train, y_train)

    print("\nK comparison:")
    print(results_df)
    print(f"\nBest K: {best_k}")

    best_model = train_model(X_train, y_train, best_k)
    results_df.to_csv(RESULTS_PATH, index=False)

    plot_accuracy_vs_k(results_df)

    evaluate_model(best_model, X_test, y_test)

    save_model(best_model)


if __name__ == "__main__":
    main()


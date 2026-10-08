import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from config import ACCURACY_PLOT_PATH, CONFUSION_MATRIX_PATH


def plot_accuracy_vs_k(results_df):
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["k"], results_df["mean_accuracy"], marker="o")
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Mean CV Accuracy")
    plt.title("Accuracy vs K")
    plt.xticks(results_df["k"])
    plt.grid(True)

    plt.savefig(ACCURACY_PLOT_PATH, bbox_inches="tight")
    plt.close()


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=model.classes_, cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_PATH, bbox_inches="tight")
    plt.close()

    return accuracy

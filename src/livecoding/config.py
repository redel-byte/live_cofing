import os


RANDOM_STATE = 22
TEST_SIZE = 0.2
K_VALUES = range(1, 10)
CV_FOLDS = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "knn_iris_pipeline.joblib")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "k_results.csv")
ACCURACY_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "accuracy_vs_k.png")
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
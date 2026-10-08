import joblib
from pathlib import Path
from config import MODEL_PATH


def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    model = joblib.load(MODEL_PATH)

    return model

import joblib

from .config import MODEL_PATH


def save_model(model):
    joblib.dump(model, MODEL_PATH)


def load_model():
    model = joblib.load(MODEL_PATH)

    return model

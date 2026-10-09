from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .save_model import load_model


app = FastAPI(title="Iris KNN API")


class IrisInput(BaseModel):
    sepal_length: float = Field(ge=0)
    sepal_width: float = Field(ge=0)
    petal_length: float = Field(ge=0)
    petal_width: float = Field(ge=0)


@app.get("/")
async def home():
    return {"message": "Iris KNN API is running"}


@lru_cache(maxsize=1)
def get_model():
    return load_model()


@app.post("/predict")
async def predict_species(data: IrisInput):
    input_df = pd.DataFrame(
        [
            {
                "sepal length (cm)": data.sepal_length,
                "sepal width (cm)": data.sepal_width,
                "petal length (cm)": data.petal_length,
                "petal width (cm)": data.petal_width,
            }
        ]
    )

    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not found. Run `uv run livecoding` before requesting predictions.",
        ) from exc

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return {
        "predicted_species": str(prediction),
        "probabilities": {
            str(species): float(probability)
            for species, probability in zip(model.classes_, probabilities)
        },
    }

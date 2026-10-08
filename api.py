import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from save_model import load_model


app = FastAPI(title="Iris KNN API")

model = load_model()


class IrisInput(BaseModel):
    sepal_length: float = Field(ge=0)
    sepal_width: float = Field(ge=0)
    petal_length: float = Field(ge=0)
    petal_width: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Iris KNN API is running"}


@app.post("/predict")
def predict_species(data: IrisInput):

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

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return {
        "predicted_species": str(prediction),
        "probabilities": {
            str(species): float(probability)
            for species, probability in zip(model.classes_, probabilities)
        },
    }

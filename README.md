# Iris KNN pipeline

This project trains and evaluates a K-nearest-neighbors classifier on the Iris
dataset, saves the trained model and evaluation artifacts, and serves
predictions through FastAPI.

## Setup and training

```powershell
uv sync --dev
uv run livecoding
```

Training writes the model, cross-validation results, accuracy chart, and
confusion matrix to `artifacts/`.

## Run the API

```powershell
uv run uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation, or
send a prediction request:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Tests

```powershell
uv run pytest -q
```

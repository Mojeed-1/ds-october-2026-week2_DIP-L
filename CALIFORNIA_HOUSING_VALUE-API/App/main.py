import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response

from App.schemas import HousingFeatures, HousingPrediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "housing_models.joblib"
MODEL_PATH = Path(os.getenv("HOUSING_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH

MODEL_NAME = os.getenv("HOUSING_MODEL_NAME", "Gradient Boosting Regressor")


def load_model_bundle() -> dict[str, Any]:
    """Load and validate the artifact produced by Training/Train.py."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run Training/Train.py to train and save the models."
        )

    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid model artifact at {MODEL_PATH}: expected a mapping.")

    preprocessor = bundle.get("preprocessor")
    models = bundle.get("models")
    if preprocessor is None or not isinstance(models, dict):
        raise ValueError(
            f"Invalid model artifact at {MODEL_PATH}: expected "
            "'preprocessor' and 'models' entries."
        )
    if MODEL_NAME not in models:
        available_models = ", ".join(sorted(models))
        raise ValueError(
            f"Model {MODEL_NAME!r} is not in {MODEL_PATH}. "
            f"Available models: {available_models}"
        )

    model = models[MODEL_NAME]
    if not callable(getattr(preprocessor, "transform", None)):
        raise ValueError("The saved preprocessor does not provide transform().")
    if not callable(getattr(model, "predict", None)):
        raise ValueError(f"The saved model {MODEL_NAME!r} does not provide predict().")

    return {"preprocessor": preprocessor, "model": model}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model_bundle()
    yield


app = FastAPI(
    title="California Housing Value API",
    description="Predict a California district's median house value.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "California Housing Value API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "loaded": hasattr(request.app.state, "model_bundle"),
    }


@app.post("/predict", response_model=HousingPrediction)
def predict(features: HousingFeatures, request: Request) -> HousingPrediction:
    model_bundle = request.app.state.model_bundle
    input_data = pd.DataFrame([features.model_dump()])
    transformed_data = model_bundle["preprocessor"].transform(input_data)
    prediction = float(model_bundle["model"].predict(transformed_data)[0])

    if prediction < 0:
        raise HTTPException(
            status_code=500,
            detail="The selected model returned a negative house-value prediction.",
        )

    return HousingPrediction(predicted_median_house_value=prediction)
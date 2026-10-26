from pathlib import Path

import joblib
import kagglehub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from mlflow.models import infer_signature
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import has_fit_parameter


DATASET_NAME = "camnugent/california-housing-prices"
TARGET = "median_house_value"
EXPERIMENT_NAME = "California Housing Prices"
N_PRICE_BANDS = 5
RANDOM_STATE = 42
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "housing_models.joblib"


def score(y_true, y_pred):
    """Return regression metrics for a set of predictions."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    dataset_path = Path(kagglehub.dataset_download(DATASET_NAME))
    data = pd.read_csv(dataset_path / "housing.csv")

    if "median_house_value" not in data.columns:
        raise ValueError("Expected target column 'median_house_value' in the dataset.")

    X = data.drop(columns=["median_house_value"])
    y = data["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Use balanced sample weights for the least common target price bands.
    train_price_bands = pd.cut(
        y_train,
        bins=N_PRICE_BANDS,
        labels=False,
        include_lowest=True,
    )
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=train_price_bands,
    )
    print("Training examples per price band:")
    print(train_price_bands.value_counts().sort_index())

    numeric_columns = X_train.select_dtypes(include="number").columns
    categorical_columns = X_train.select_dtypes(exclude="number").columns
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "MLP Neural Network Regressor": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
    }

    mlflow.set_tracking_uri("http://localhost:8100")
    client = MlflowClient()
    experiment = mlflow.set_experiment("California Housing value")
    mlflow.set_experiment_tags(
        {
            "project": "California-Housing-value",
            "dataset": "california-housing",
            "task": "continuous value prediction",
            "primary_metrics": "r2, rmse, mae",
        }
    )
    print(f"\nMLflow experiment: {experiment.name} (ID: {experiment.experiment_id})")

    results = {}
    for model_name, model in models.items():
        accepts_sample_weight = has_fit_parameter(model, "sample_weight")

        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("algorithm", model_name)
            mlflow.log_param("target_price_bands", N_PRICE_BANDS)
            mlflow.log_param("uses_sample_weight", accepts_sample_weight)

            if accepts_sample_weight:
                model.fit(
                    X_train_processed,
                    y_train,
                    sample_weight=sample_weights,
                )
            else:
                print(
                    f"Note: {model_name} does not support sample_weight; "
                    "fitting without price-band weights."
                )
                model.fit(X_train_processed, y_train)

            train_predictions = model.predict(X_train_processed)
            test_predictions = model.predict(X_test_processed)
            train_metrics = score(y_train, train_predictions)
            test_metrics = score(y_test, test_predictions)
            results[model_name] = {
                "train": train_metrics,
                "test": test_metrics,
            }

            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

            model_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )
            signature = infer_signature(
                X_train.head(),
                model_pipeline.predict(X_train.head()),
            )
            registered_model_name = (
                "CaliforniaHousing"
                + model_name.replace(" ", "").replace("-", "")
            )
            mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                name="model",
                registered_model_name=registered_model_name,
                serialization_format="cloudpickle",
                signature=signature,
                input_example=X_train.head(),
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=test_predictions, ax=ax)
            ax.set_title(f"{model_name}: actual vs. predicted")
            ax.set_xlabel("Actual house value")
            ax.set_ylabel("Predicted house value")
            mlflow.log_figure(fig, "actual_vs_predicted.png")
            plt.close(fig)

            print(f"\n{model_name}")
            print(f"  Train metrics: {train_metrics}")
            print(f"  Test metrics:  {test_metrics}")

    artifact = {
        "preprocessor": preprocessor,
        "models": models,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(
        "\nCompare train and test scores: a much better training score "
        "can indicate overfitting."
    )
    return results


if __name__ == "__main__":
    main()
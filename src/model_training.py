
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_PATH,
    METRICS_PATH,
    METADATA_PATH,
)

from .feature_engineering import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    prepare_training_frame,
)


def make_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        ),
    ])


def train_and_evaluate(
    raw: pd.DataFrame,
    model_dir=None,
    test_size=TEST_SIZE
):

    X, y, timestamps, cleaned = prepare_training_frame(raw)

    # Chronological split using unique timestamps
    unique_times = np.sort(timestamps.unique())

    if len(unique_times) < 10:
        raise ValueError(
            "At least 10 distinct timestamps are required."
        )

    split_index = int(len(unique_times) * (1 - test_size))
    split_index = max(1, min(split_index, len(unique_times) - 1))

    cutoff = pd.Timestamp(unique_times[split_index])

    train_mask = timestamps < cutoff
    test_mask = timestamps >= cutoff

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    if X_train.empty or X_test.empty:
        raise ValueError("Chronological split produced an empty set.")

    # Models selected for comparison
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    fitted_models = {}

    for name, estimator in models.items():

        print(f"Training {name}...")

        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])

        start = time.time()

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        elapsed = round(time.time() - start, 2)

        results.append({
            "model": name,
            "MAE": float(
                mean_absolute_error(y_test, predictions)
            ),
            "RMSE": float(
                np.sqrt(mean_squared_error(y_test, predictions))
            ),
            "R2": float(
                r2_score(y_test, predictions)
            ),
            "training_seconds": elapsed,
        })

        fitted_models[name] = pipeline

        print(f"{name} completed in {elapsed} seconds.")

    metrics = pd.DataFrame(results).sort_values(
        ["MAE", "RMSE"]
    ).reset_index(drop=True)

    best_name = str(metrics.iloc[0]["model"])
    best_pipeline = fitted_models[best_name]

    # Save model and evaluation files
    if model_dir is None:
        model_path = MODEL_PATH
        metrics_path = METRICS_PATH
        metadata_path = METADATA_PATH
    else:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "traffic_pipeline.joblib"
        metrics_path = model_dir / "model_metrics.json"
        metadata_path = model_dir / "model_metadata.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, model_path, compress=1)

    metrics.to_json(
        metrics_path,
        orient="records",
        indent=2
    )

    metadata = {
        "best_model": best_name,
        "split_cutoff": str(cutoff),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cleaned_rows": int(len(cleaned)),
        "features": list(X.columns),
        "target_thresholds": {
            "low_max": float(y_train.quantile(1 / 3)),
            "moderate_max": float(y_train.quantile(2 / 3)),
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    predictions = best_pipeline.predict(X_test)

    return {
        "best_model": best_name,
        "pipeline": best_pipeline,
        "metrics": metrics,
        "metadata": metadata,
        "X_test": X_test,
        "y_test": y_test,
        "test_predictions": predictions,
        "cleaned": cleaned,
        "model_path": str(model_path),
    }

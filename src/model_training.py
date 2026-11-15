
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .feature_engineering import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    prepare_training_frame
)

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_PATH,
    METRICS_PATH,
    METADATA_PATH
)


# --------------------------------------------------
# PREPROCESSING PIPELINE
# --------------------------------------------------

def make_preprocessor():

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )


# --------------------------------------------------
# MODEL TRAINING AND EVALUATION
# --------------------------------------------------

def train_and_evaluate(
    raw: pd.DataFrame,
    model_dir=None,
    test_size=TEST_SIZE
):

    started_total = time.time()

    # Prepare dataset
    X, y, timestamps, cleaned = prepare_training_frame(raw)

    if len(X) == 0:
        raise ValueError("No valid training records found.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    # --------------------------------------------------
    # CHRONOLOGICAL TRAIN-TEST SPLIT
    # --------------------------------------------------

    unique_times = np.array(
        sorted(timestamps.dropna().unique())
    )

    if len(unique_times) < 10:
        raise ValueError(
            "At least 10 distinct timestamps are required."
        )

    split_index = int(
        len(unique_times) * (1 - test_size)
    )

    split_index = max(
        1,
        min(len(unique_times) - 1, split_index)
    )

    cutoff = pd.Timestamp(unique_times[split_index])

    train_mask = timestamps < cutoff
    test_mask = timestamps >= cutoff

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Chronological split produced an empty dataset."
        )

    # --------------------------------------------------
    # MODELS
    # --------------------------------------------------

    models = {

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=2,
            random_state=RANDOM_STATE
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=2,
            random_state=RANDOM_STATE
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        )
    }

    # --------------------------------------------------
    # TRAIN AND EVALUATE EACH MODEL
    # --------------------------------------------------

    results = []
    fitted = {}

    for name, estimator in models.items():

        model_started = time.time()

        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test, predictions
        )

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(
            y_test, predictions
        )

        results.append({
            "model": name,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "training_seconds": round(
                time.time() - model_started, 2
            )
        })

        fitted[name] = pipeline

    # --------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------

    metrics = pd.DataFrame(results)

    metrics = metrics.sort_values(
        ["MAE", "RMSE"],
        ascending=True
    ).reset_index(drop=True)

    best_name = str(
        metrics.iloc[0]["model"]
    )

    best_pipeline = fitted[best_name]

    # --------------------------------------------------
    # SAVE MODEL AND RESULTS
    # --------------------------------------------------

    if model_dir is None:

        model_path = MODEL_PATH
        metrics_path = METRICS_PATH
        metadata_path = METADATA_PATH

    else:

        from pathlib import Path

        target_dir = Path(model_dir)
        target_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = target_dir / "traffic_pipeline.joblib"
        metrics_path = target_dir / "model_metrics.json"
        metadata_path = target_dir / "model_metadata.json"

    # Ensure output directories exist
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_pipeline,
        model_path,
        compress=1
    )

    metrics.to_json(
        metrics_path,
        orient="records",
        indent=2
    )

    # --------------------------------------------------
    # MODEL METADATA
    # --------------------------------------------------

    metadata = {

        "best_model": best_name,

        "split_cutoff": str(cutoff),

        "train_rows": int(len(X_train)),

        "test_rows": int(len(X_test)),

        "cleaned_rows": int(len(cleaned)),

        "train_timestamp_start": str(
            timestamps.loc[train_mask].min()
        ),

        "train_timestamp_end": str(
            timestamps.loc[train_mask].max()
        ),

        "test_timestamp_start": str(
            timestamps.loc[test_mask].min()
        ),

        "test_timestamp_end": str(
            timestamps.loc[test_mask].max()
        ),

        "features": list(X.columns),

        "target_thresholds": {

            "low_max": float(
                y_train.quantile(1 / 3)
            ),

            "moderate_max": float(
                y_train.quantile(2 / 3)
            )
        }
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    # --------------------------------------------------
    # FINAL PREDICTIONS
    # --------------------------------------------------

    test_predictions = best_pipeline.predict(
        X_test
    )

    total_seconds = round(
        time.time() - started_total, 2
    )

    return {

        "best_model": best_name,

        "pipeline": best_pipeline,

        "metrics": metrics,

        "metadata": metadata,

        "X_test": X_test,

        "y_test": y_test,

        "test_predictions": test_predictions,

        "cleaned": cleaned,

        "model_path": str(model_path),

        "training_seconds": total_seconds
    }

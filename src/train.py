"""Train and evaluate tree-based models on the official OPSSAT-AD split."""

import json
import platform
import copy
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_loader import ENGINEERED_FEATURE_COLUMNS, TARGET_COLUMN


RANDOM_STATE = 42
DEFAULT_CV_SPLITS = 3
DEFAULT_TUNING_ITERATIONS = 12
PROCESSED_TRAIN_FILENAME = "train_processed.csv"
PROCESSED_TEST_FILENAME = "test_processed.csv"
PREPROCESSOR_FILENAME = "preprocessing_pipeline.joblib"


def _validate_processed_table(
    frame: pd.DataFrame,
    expected_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate persisted Stage 3 features and keep the target out of X."""
    expected_columns = [*ENGINEERED_FEATURE_COLUMNS, TARGET_COLUMN]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"{expected_name} must contain the 18 benchmark features followed by "
            f"{TARGET_COLUMN!r}; got {frame.columns.tolist()}."
        )
    if frame.isna().any().any():
        raise ValueError(f"{expected_name} still contains missing values.")
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{expected_name} must preserve binary labels 0 and 1.")

    X = frame.loc[:, list(ENGINEERED_FEATURE_COLUMNS)].copy()
    y = frame[TARGET_COLUMN].copy().astype(int)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"{expected_name} contains non-finite feature values.")
    if TARGET_COLUMN in X.columns or "train" in X.columns:
        raise AssertionError("Target or split leakage found in model input columns.")
    return X, y


def load_processed_split(
    processed_dir: Union[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    """Load Stage 3 outputs and fitted preprocessor without creating a new split."""
    directory = Path(processed_dir)
    train_path = directory / PROCESSED_TRAIN_FILENAME
    test_path = directory / PROCESSED_TEST_FILENAME
    preprocessor_path = directory / PREPROCESSOR_FILENAME
    for path in (train_path, test_path, preprocessor_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Stage 3 output is missing: {path}. "
                "Run notebooks/03_data_preprocessing.ipynb first."
            )

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    X_train, y_train = _validate_processed_table(train_frame, train_path.name)
    X_test, y_test = _validate_processed_table(test_frame, test_path.name)
    if X_train.empty or X_test.empty:
        raise ValueError("Both official train and test files must contain rows.")
    if set(y_train.unique()) != {0, 1} or set(y_test.unique()) != {0, 1}:
        raise ValueError("Both official splits must contain normal and anomalous labels.")

    preprocessor = joblib.load(preprocessor_path)
    feature_names = tuple(getattr(preprocessor, "feature_names_in_", ()))
    if feature_names != ENGINEERED_FEATURE_COLUMNS:
        raise ValueError(
            "The saved Stage 3 preprocessing pipeline has unexpected feature names. "
            f"Expected {ENGINEERED_FEATURE_COLUMNS}, received {feature_names}."
        )
    transformed_train = preprocessor.transform(X_train)
    transformed_test = preprocessor.transform(X_test)
    if not np.allclose(transformed_train, X_train.to_numpy()) or not np.allclose(
        transformed_test, X_test.to_numpy()
    ):
        raise ValueError(
            "Stage 3 pipeline changes already-processed values. Confirm the processed "
            "CSVs and fitted pipeline were produced together."
        )

    return X_train, X_test, y_train, y_test, preprocessor


def _base_models(random_state: int) -> dict:
    """Return small, reproducible tree-model baselines."""
    return {
        "Decision Tree baseline": DecisionTreeClassifier(
            random_state=random_state,
        ),
        "Random Forest baseline": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=1,
        ),
        "XGBoost baseline": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
        ),
    }


def _metric_values(y_true: pd.Series, predictions, probabilities) -> dict:
    """Calculate class-sensitive and threshold-independent binary metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_anomaly": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_anomaly": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_anomaly": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def train_and_evaluate(
    processed_dir: Union[str, Path],
    artifact_dir: Union[str, Path],
    random_state: int = RANDOM_STATE,
    cv_splits: int = DEFAULT_CV_SPLITS,
    tuning_iterations: int = DEFAULT_TUNING_ITERATIONS,
) -> dict:
    """Train baselines, tune XGBoost on training-only CV, and evaluate the test set.

    The official test rows are only used after model selection, for final holdout
    reporting. Average precision is the CV selection metric because anomalies are
    the minority class. No resampling is used.
    """
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    if tuning_iterations < 1:
        raise ValueError("tuning_iterations must be at least 1.")

    X_train, X_test, y_train, y_test, preprocessor = load_processed_split(processed_dir)
    minority_count = int(y_train.value_counts().min())
    if cv_splits > minority_count:
        raise ValueError(
            f"cv_splits={cv_splits} exceeds the smallest training class "
            f"({minority_count})."
        )
    cv = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state
    )
    models = _base_models(random_state)
    cv_results = {}
    test_results = {}
    confusion_matrices = {}

    # Cross-validation fits each baseline using only folds from the training set.
    for model_name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
        )
        cv_results[model_name] = {
            "average_precision_mean": float(scores.mean()),
            "average_precision_std": float(scores.std()),
            "fold_scores": [float(score) for score in scores],
        }

    # The compact randomized search samples combinations of all five requested
    # XGBoost parameters, with CV restricted to the training data.
    tuning_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
    )
    parameter_space = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    search = RandomizedSearchCV(
        estimator=tuning_model,
        param_distributions=parameter_space,
        n_iter=tuning_iterations,
        scoring="average_precision",
        cv=cv,
        refit=True,
        random_state=random_state,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    best_xgboost = search.best_estimator_
    cv_results["Tuned XGBoost"] = {
        "average_precision_mean": float(search.best_score_),
        "average_precision_std": float(
            search.cv_results_["std_test_score"][search.best_index_]
        ),
        "fold_scores": [
            float(search.cv_results_[f"split{fold}_test_score"][search.best_index_])
            for fold in range(cv_splits)
        ],
    }

    final_models = {**models, "Tuned XGBoost": best_xgboost}
    for model_name, model in final_models.items():
        # Test data is first used here, after CV selection is complete.
        if model_name != "Tuned XGBoost":
            model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        test_results[model_name] = _metric_values(
            y_test, predictions, probabilities
        )
        matrix = pd.crosstab(
            pd.Series(y_test.to_numpy(), name="actual"),
            pd.Series(predictions, name="predicted"),
        ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        confusion_matrices[model_name] = matrix

    # Preserve feature labels through the preprocessing transform so that the
    # serialized inference pipeline matches the model's training feature names.
    preprocessor.set_output(transform="pandas")
    final_pipeline = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("classifier", best_xgboost),
        ]
    )
    pipeline_predictions = final_pipeline.predict(X_test)
    if not np.array_equal(pipeline_predictions, best_xgboost.predict(X_test)):
        raise AssertionError("Saved inference pipeline changed tuned model predictions.")

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "xgboost_model.joblib"
    model_filenames = {
        "Decision Tree baseline": "decision_tree_model.joblib",
        "Random Forest baseline": "random_forest_model.joblib",
        "XGBoost baseline": "xgboost_baseline_model.joblib",
        "Tuned XGBoost": "xgboost_model.joblib",
    }
    preprocessing_path = output / "preprocessing_pipeline.joblib"
    feature_info_path = output / "feature_names.json"
    results_path = output / "model_comparison.csv"
    metadata_path = output / "model_metadata.json"

    saved_model_paths = {}
    for model_name, model in final_models.items():
        inference_pipeline = Pipeline(
            [
                ("preprocessing", copy.deepcopy(preprocessor)),
                ("classifier", model),
            ]
        )
        predictions = inference_pipeline.predict(X_test)
        if not np.array_equal(predictions, model.predict(X_test)):
            raise AssertionError(
                f"Saved preprocessing pipeline changes {model_name} predictions."
            )
        model_path_for_name = output / model_filenames[model_name]
        joblib.dump(inference_pipeline, model_path_for_name, compress=3)
        saved_model_paths[model_name] = model_path_for_name

    joblib.dump(preprocessor, preprocessing_path, compress=3)
    feature_info = {
        "features": list(ENGINEERED_FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "excluded_inputs": ["anomaly", "train", "segment", "channel", "sampling"],
        "feature_count": len(ENGINEERED_FEATURE_COLUMNS),
        "class_mapping": {"0": "Normal", "1": "Anomalous"},
    }
    feature_info_path.write_text(json.dumps(feature_info, indent=2), encoding="utf-8")

    rows = []
    for model_name in final_models:
        row = {
            "model": model_name,
            "cv_average_precision_mean": cv_results[model_name][
                "average_precision_mean"
            ],
            "cv_average_precision_std": cv_results[model_name][
                "average_precision_std"
            ],
            **{f"test_{key}": value for key, value in test_results[model_name].items()},
        }
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(results_path, index=False)

    model_configurations = {
        name: model.get_params(deep=False)
        for name, model in final_models.items()
    }
    metadata = {
        "selection_metric": "average_precision",
        "cross_validation": {
            "method": "StratifiedKFold",
            "folds": cv_splits,
            "shuffle": True,
            "random_state": random_state,
            "training_rows_only": True,
        },
        "xgboost_tuning": {
            "search": "RandomizedSearchCV",
            "iterations": tuning_iterations,
            "parameter_space": parameter_space,
            "best_parameters": search.best_params_,
            "best_cv_average_precision": float(search.best_score_),
        },
        "official_split": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_class_counts": {
                str(key): int(value) for key, value in y_train.value_counts().sort_index().items()
            },
            "test_class_counts": {
                str(key): int(value) for key, value in y_test.value_counts().sort_index().items()
            },
        },
        "models": model_configurations,
        "cv_results": cv_results,
        "test_results": test_results,
        "confusion_matrices": {
            name: matrix.to_dict() for name, matrix in confusion_matrices.items()
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "artifacts": {
            "models": {
                name: path.name for name, path in saved_model_paths.items()
            },
            "preprocessing_pipeline": preprocessing_path.name,
            "feature_information": feature_info_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": final_models,
        "cv_results": cv_results,
        "test_results": test_results,
        "confusion_matrices": confusion_matrices,
        "best_params": search.best_params_,
        "best_cv_average_precision": float(search.best_score_),
        "comparison": comparison,
        "artifact_paths": {
            **{
                f"model_{name.lower().replace(' ', '_')}": path
                for name, path in saved_model_paths.items()
            },
            "model": saved_model_paths["Tuned XGBoost"],
            "preprocessing_pipeline": preprocessing_path,
            "feature_information": feature_info_path,
            "comparison": results_path,
            "metadata": metadata_path,
        },
    }


def main() -> None:
    """Train the Stage 4 models from project-relative Stage 3 artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    result = train_and_evaluate(
        processed_dir=project_root / "data" / "processed",
        artifact_dir=project_root / "models",
    )
    print(result["comparison"].to_string(index=False))
    print("\nBest XGBoost parameters (selected using training-only CV):")
    print(json.dumps(result["best_params"], indent=2))
    for artifact_name, artifact_path in result["artifact_paths"].items():
        print(f"{artifact_name}: {artifact_path}")


if __name__ == "__main__":
    main()

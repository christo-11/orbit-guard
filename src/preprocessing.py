"""Reproducible preprocessing for the official OPSSAT-AD benchmark split."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

from src.data_loader import (
    ENGINEERED_FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)


IDENTIFIER_COLUMNS = ("segment",)
METADATA_COLUMNS = ("channel", "sampling")
EXCLUDED_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    *METADATA_COLUMNS,
    TARGET_COLUMN,
    SPLIT_COLUMN,
)
TRAIN_VALUE = 1
TEST_VALUE = 0


@dataclass
class PreprocessedData:
    """Prepared benchmark split and diagnostics, with labels kept unchanged."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    imputer: SimpleImputer
    outlier_summary: pd.DataFrame
    leakage_check_passed: bool


def inspect_dataset(frame: pd.DataFrame) -> dict:
    """Summarize missing values and duplicates without changing the source table."""
    required_columns = {*ENGINEERED_FEATURE_COLUMNS, TARGET_COLUMN, SPLIT_COLUMN}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    duplicate_segment_count = (
        int(frame["segment"].duplicated().sum()) if "segment" in frame.columns else None
    )
    return {
        "shape": frame.shape,
        "missing_by_column": frame.isna().sum(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_feature_rows": int(
            frame.loc[:, list(ENGINEERED_FEATURE_COLUMNS)].duplicated().sum()
        ),
        "duplicate_segment_ids": duplicate_segment_count,
    }


def _validate_dataset(frame: pd.DataFrame) -> None:
    """Fail clearly if the benchmark schema, labels, or split is unexpected."""
    required_columns = {*ENGINEERED_FEATURE_COLUMNS, TARGET_COLUMN, SPLIT_COLUMN}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("The feature dataset is empty.")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("The target contains missing values; labels will not be imputed.")
    if frame[SPLIT_COLUMN].isna().any():
        raise ValueError("The official split column contains missing values.")
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Expected the original binary anomaly labels 0 and 1.")
    if set(frame[SPLIT_COLUMN].unique()) != {TRAIN_VALUE, TEST_VALUE}:
        raise ValueError("Expected official split values train=1 and train=0.")
    if "segment" in frame.columns and frame["segment"].duplicated().any():
        raise ValueError("Segment identifiers are not unique; refusing ambiguous rows.")

    non_numeric = [
        column
        for column in ENGINEERED_FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Expected numeric benchmark features, found: {non_numeric}")


def _training_outlier_summary(X_train: pd.DataFrame) -> pd.DataFrame:
    """Count training values beyond 1.5×IQR fences without removing any values."""
    first_quartile = X_train.quantile(0.25)
    third_quartile = X_train.quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    lower_fence = first_quartile - 1.5 * interquartile_range
    upper_fence = third_quartile + 1.5 * interquartile_range
    outlier_mask = X_train.lt(lower_fence) | X_train.gt(upper_fence)
    return pd.DataFrame(
        {
            "potential_outlier_count": outlier_mask.sum().astype(int),
            "percent_of_training_rows": (outlier_mask.mean() * 100).round(2),
        }
    ).sort_values("potential_outlier_count", ascending=False)


def prepare_dataset(
    frame: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
) -> PreprocessedData:
    """Prepare numeric features using the official split and training-only imputation.

    The source frame is not modified. The supplied target values are copied unchanged.
    No scaling, resampling, feature removal, or outlier removal is performed.
    """
    _validate_dataset(frame)

    # Normalize non-finite numeric values to missing so the training-fitted imputer
    # handles them consistently with ordinary NaN values.
    feature_values = frame.loc[:, list(ENGINEERED_FEATURE_COLUMNS)].copy()
    feature_values = feature_values.replace([np.inf, -np.inf], np.nan)

    train_mask = frame[SPLIT_COLUMN].eq(TRAIN_VALUE)
    test_mask = frame[SPLIT_COLUMN].eq(TEST_VALUE)
    X_train_raw = feature_values.loc[train_mask].copy()
    X_test_raw = feature_values.loc[test_mask].copy()
    y_train = frame.loc[train_mask, TARGET_COLUMN].copy().rename(TARGET_COLUMN)
    y_test = frame.loc[test_mask, TARGET_COLUMN].copy().rename(TARGET_COLUMN)

    if X_train_raw.empty or X_test_raw.empty:
        raise ValueError("Both official training and test partitions must contain rows.")
    if not X_train_raw.index.intersection(X_test_raw.index).empty:
        raise ValueError("Training and test rows overlap.")

    # IQR fences are measured on training data only, and the values are retained.
    outlier_summary = _training_outlier_summary(X_train_raw)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_values = imputer.fit_transform(X_train_raw)
    X_test_values = imputer.transform(X_test_raw)
    X_train = pd.DataFrame(
        X_train_values, columns=ENGINEERED_FEATURE_COLUMNS, index=X_train_raw.index
    )
    X_test = pd.DataFrame(
        X_test_values, columns=ENGINEERED_FEATURE_COLUMNS, index=X_test_raw.index
    )

    leakage_check_passed = not set(X_train.columns).intersection(EXCLUDED_COLUMNS)
    if not leakage_check_passed:
        raise AssertionError("An identifier, metadata, target, or split column leaked into X.")
    if tuple(X_train.columns) != ENGINEERED_FEATURE_COLUMNS:
        raise AssertionError("The final feature columns differ from the benchmark allowlist.")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise AssertionError("Missing values remain after training-fitted imputation.")
    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(X_test.to_numpy()).all():
        raise AssertionError("Non-finite values remain after preprocessing.")

    prepared = PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        imputer=imputer,
        outlier_summary=outlier_summary,
        leakage_check_passed=leakage_check_passed,
    )

    if output_dir is not None:
        save_preprocessed_data(prepared, output_dir)
    return prepared


def save_preprocessed_data(
    prepared: PreprocessedData,
    output_dir: Union[str, Path],
) -> Path:
    """Save feature/target tables and the fitted imputation values for reuse."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    train_output = prepared.X_train.copy()
    train_output[TARGET_COLUMN] = prepared.y_train
    test_output = prepared.X_test.copy()
    test_output[TARGET_COLUMN] = prepared.y_test
    train_output.to_csv(destination / "train_processed.csv", index=False)
    test_output.to_csv(destination / "test_processed.csv", index=False)
    joblib.dump(prepared.imputer, destination / "preprocessing_pipeline.joblib")

    metadata = {
        "features": list(ENGINEERED_FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "official_split": {
            "column": SPLIT_COLUMN,
            "training_value": TRAIN_VALUE,
            "test_value": TEST_VALUE,
        },
        "excluded_columns": list(EXCLUDED_COLUMNS),
        "imputation": {
            "strategy": "median",
            "fit_partition": "training only",
            "pipeline_file": "preprocessing_pipeline.joblib",
            "training_medians": {
                feature: float(value)
                for feature, value in zip(
                    ENGINEERED_FEATURE_COLUMNS, prepared.imputer.statistics_
                )
            },
        },
        "scaling": "not applied",
        "outlier_policy": "diagnosed on training data using 1.5xIQR; values retained",
        "resampling": "not applied",
        "randomness": "none; official split and deterministic median imputation",
    }
    with (destination / "preprocessing_metadata.json").open(
        "w", encoding="utf-8"
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return destination

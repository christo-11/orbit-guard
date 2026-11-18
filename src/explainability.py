"""SHAP explanations for the saved baseline XGBoost telemetry classifier."""

import json
from pathlib import Path
from typing import Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data_loader import ENGINEERED_FEATURE_COLUMNS, TARGET_COLUMN


BASELINE_MODEL_FILENAME = "xgboost_baseline_model.joblib"
PREPROCESSOR_FILENAME = "preprocessing_pipeline.joblib"
FEATURE_INFO_FILENAME = "feature_names.json"
TRAIN_FILENAME = "train_processed.csv"
TEST_FILENAME = "test_processed.csv"
DEFAULT_RANDOM_STATE = 42


def _require_file(path: Path) -> None:
    """Raise an actionable error when a Stage 3/4 input is not present."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Stage 6 input is missing: {path}. "
            "Run Stages 3 and 4 first, then ensure the baseline XGBoost artifact "
            "has been saved."
        )


def _normalize_shap_output(
    explanation: shap.Explanation,
    row_count: int,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize SHAP binary-output variants to class-1 values and base values."""
    values = explanation.values
    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError(f"Expected two binary SHAP outputs, got {len(values)}.")
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[0] == row_count and values.shape[1] == feature_count:
            if values.shape[2] != 2:
                raise ValueError(f"Unexpected SHAP output axis: {values.shape}.")
            values = values[:, :, 1]
        elif values.shape[0] == 2 and values.shape[1:] == (
            row_count,
            feature_count,
        ):
            values = values[1]
        else:
            raise ValueError(f"Unexpected SHAP values shape: {values.shape}.")
    if values.shape != (row_count, feature_count):
        raise ValueError(
            f"Expected SHAP values with shape {(row_count, feature_count)}, "
            f"got {values.shape}."
        )

    base_values = np.asarray(explanation.base_values)
    if base_values.ndim == 0:
        base_values = np.full(row_count, float(base_values))
    elif base_values.ndim == 2:
        if base_values.shape == (row_count, 2):
            base_values = base_values[:, 1]
        elif base_values.shape == (2, row_count):
            base_values = base_values[1]
        else:
            raise ValueError(f"Unexpected SHAP base values shape: {base_values.shape}.")
    base_values = base_values.reshape(-1)
    if base_values.shape != (row_count,):
        raise ValueError(
            f"Expected {row_count} SHAP base values, got {base_values.shape}."
        )
    return values, base_values


def _select_explanation_rows(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    max_rows: int,
    random_state: int,
) -> tuple[np.ndarray, dict]:
    """Choose a deterministic, class-balanced test sample including local examples."""
    actual_classes = y_test.to_numpy()
    normal_candidates = np.flatnonzero((actual_classes == 0) & (predictions == 0))
    anomalous_candidates = np.flatnonzero(
        (actual_classes == 1) & (predictions == 1)
    )
    if not len(normal_candidates) or not len(anomalous_candidates):
        raise ValueError(
            "A correctly classified normal and anomalous test example are both "
            "required for the requested waterfall plots."
        )
    selected_positions = {
        "normal": int(normal_candidates[0]),
        "anomalous": int(anomalous_candidates[0]),
    }

    class_values = actual_classes
    class_counts = {value: int(np.sum(class_values == value)) for value in (0, 1)}
    per_class = min(max_rows // 2, min(class_counts.values()))
    if per_class < 1:
        raise ValueError("The test set is too small to sample both binary classes.")

    chosen: list[int] = []
    rng = np.random.default_rng(random_state)
    for class_value, key in ((0, "normal"), (1, "anomalous")):
        required_position = selected_positions[key]
        candidates = np.flatnonzero(class_values == class_value)
        remaining = candidates[candidates != required_position]
        additional_count = min(per_class - 1, len(remaining))
        if additional_count:
            chosen.extend(
                rng.choice(remaining, size=additional_count, replace=False).tolist()
            )
        chosen.append(required_position)
    positions = np.asarray(sorted(set(chosen)), dtype=int)
    if len(positions) > max_rows:
        raise AssertionError("Explanation subset exceeds its configured row limit.")
    if not np.any(class_values[positions] == 0) or not np.any(
        class_values[positions] == 1
    ):
        raise AssertionError("Explanation subset must contain both actual classes.")
    return positions, selected_positions


def _make_local_explanation(
    position: int,
    kind: str,
    feature_values: pd.DataFrame,
    shap_values: np.ndarray,
    base_values: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    y_test: pd.Series,
) -> tuple[shap.Explanation, dict]:
    """Build a class-1 probability Explanation and a ranked contribution summary."""
    row_values = shap_values[position]
    explanation = shap.Explanation(
        values=row_values,
        base_values=float(base_values[position]),
        data=feature_values.iloc[position].to_numpy(dtype=float),
        feature_names=list(feature_values.columns),
    )
    ranked_indices = np.argsort(np.abs(row_values))[::-1]
    contributions = [
        {
            "feature": str(feature_values.columns[index]),
            "feature_value": float(feature_values.iloc[position, index]),
            "shap_value": float(row_values[index]),
            "direction": (
                "increases the model's anomaly probability"
                if row_values[index] > 0
                else "decreases the model's anomaly probability"
                if row_values[index] < 0
                else "does not change the model's anomaly probability"
            ),
        }
        for index in ranked_indices
    ]
    summary = {
        "kind": kind,
        "test_row_position": int(position),
        "actual_class": int(y_test.iloc[position]),
        "predicted_class": int(predictions[position]),
        "predicted_label": "Anomalous" if predictions[position] == 1 else "Normal",
        "predicted_anomaly_probability": float(probabilities[position]),
        "base_probability": float(base_values[position]),
        "shap_probability_sum": float(row_values.sum()),
        "reconstructed_anomaly_probability": float(
            base_values[position] + row_values.sum()
        ),
        "top_contributions": contributions[:5],
    }
    return explanation, summary


def explain_baseline_xgboost(
    model_dir: Union[str, Path],
    processed_dir: Union[str, Path],
    figures_dir: Union[str, Path],
    summary_dir: Union[str, Path],
    raw_dataset_path: Union[str, Path, None] = None,
    background_rows: int = 100,
    explanation_rows: int = 128,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Explain the saved baseline XGBoost model without retraining or changing it."""
    if background_rows < 1 or explanation_rows < 2:
        raise ValueError("At least one background row and two explanation rows are required.")

    model_directory = Path(model_dir)
    processed_directory = Path(processed_dir)
    feature_info_path = model_directory / FEATURE_INFO_FILENAME
    model_path = model_directory / BASELINE_MODEL_FILENAME
    preprocessor_path = model_directory / PREPROCESSOR_FILENAME
    train_path = processed_directory / TRAIN_FILENAME
    test_path = processed_directory / TEST_FILENAME
    for path in (
        feature_info_path,
        model_path,
        preprocessor_path,
        train_path,
        test_path,
    ):
        _require_file(path)

    feature_info = json.loads(feature_info_path.read_text(encoding="utf-8"))
    feature_names = feature_info.get("features")
    if feature_names != list(ENGINEERED_FEATURE_COLUMNS):
        raise ValueError("Saved feature names do not match the Stage 3 feature contract.")
    excluded = set(feature_info.get("excluded_inputs", []))
    if not {"anomaly", "train"}.issubset(excluded):
        raise ValueError("Feature metadata must explicitly exclude anomaly and train.")
    if feature_info.get("target") != TARGET_COLUMN:
        raise ValueError(f"Expected {TARGET_COLUMN!r} as the saved target name.")

    model_pipeline = joblib.load(model_path)
    saved_preprocessor = joblib.load(preprocessor_path)
    if not hasattr(model_pipeline, "named_steps"):
        raise TypeError("Baseline XGBoost artifact is not a preprocessing/model pipeline.")
    if set(model_pipeline.named_steps) != {"preprocessing", "classifier"}:
        raise ValueError(
            "Expected the baseline pipeline steps 'preprocessing' and 'classifier'."
        )
    pipeline_preprocessor = model_pipeline.named_steps["preprocessing"]
    classifier = model_pipeline.named_steps["classifier"]
    if not hasattr(classifier, "get_booster"):
        raise TypeError("The saved baseline classifier is not a compatible XGBoost model.")
    for description, transformer in (
        ("saved standalone preprocessor", saved_preprocessor),
        ("preprocessor embedded in model", pipeline_preprocessor),
    ):
        fitted_names = tuple(getattr(transformer, "feature_names_in_", ()))
        if fitted_names != ENGINEERED_FEATURE_COLUMNS:
            raise ValueError(
                f"The {description} does not match the saved feature names: "
                f"{fitted_names}."
            )
    classifier_features = tuple(
        getattr(classifier, "feature_names_in_", ())
    )
    if classifier_features != ENGINEERED_FEATURE_COLUMNS:
        raise ValueError(
            "Baseline XGBoost feature names do not match the saved feature names."
        )

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    expected_columns = [*ENGINEERED_FEATURE_COLUMNS, TARGET_COLUMN]
    for description, frame in (("training", train_frame), ("test", test_frame)):
        if frame.columns.tolist() != expected_columns:
            raise ValueError(
                f"The {description} file must contain only the saved features and "
                f"{TARGET_COLUMN}; found {frame.columns.tolist()}."
            )
        if frame.isna().any().any():
            raise ValueError(f"The {description} file contains missing values.")
    X_train = train_frame.loc[:, feature_names]
    X_test = test_frame.loc[:, feature_names]
    y_test = test_frame[TARGET_COLUMN].astype(int)
    if set(y_test.unique()) != {0, 1}:
        raise ValueError("The official test set must contain both labels 0 and 1.")

    # Apply both stored versions of Stage 3 preprocessing and require equivalence.
    X_background_all = saved_preprocessor.transform(X_train)
    X_model_background = pipeline_preprocessor.transform(X_train)
    X_test_preprocessed = saved_preprocessor.transform(X_test)
    X_model_test = pipeline_preprocessor.transform(X_test)
    if not np.allclose(X_background_all, X_model_background):
        raise ValueError("The Stage 3 and embedded preprocessing pipelines differ.")
    if not np.allclose(X_test_preprocessed, X_model_test):
        raise ValueError("The two preprocessing paths differ on test data.")

    X_background_all = pd.DataFrame(
        X_background_all, columns=feature_names, index=X_train.index
    )
    X_test_preprocessed = pd.DataFrame(
        X_test_preprocessed, columns=feature_names, index=X_test.index
    )
    if set(X_background_all.columns).intersection({"anomaly", "train"}):
        raise AssertionError("Target or split leaked into the SHAP background data.")

    sample_count = min(background_rows, len(X_background_all))
    background = X_background_all.sample(
        n=sample_count, random_state=random_state
    )
    # Probabilities are explained so positive/negative SHAP values have an
    # intuitive interpretation as raising/lowering P(anomaly=1).
    explainer = shap.TreeExplainer(
        classifier,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )

    all_test_predictions = np.asarray(model_pipeline.predict(X_test), dtype=int)
    all_test_probabilities = np.asarray(model_pipeline.predict_proba(X_test))[:, 1]
    classifier_predictions = np.asarray(
        classifier.predict(X_model_test), dtype=int
    )
    classifier_probabilities = np.asarray(
        classifier.predict_proba(X_model_test)
    )[:, 1]
    if not np.array_equal(all_test_predictions, classifier_predictions) or not np.allclose(
        all_test_probabilities, classifier_probabilities, atol=1e-7
    ):
        raise AssertionError(
            "The saved inference pipeline and exact preprocessor/model path disagree."
        )

    explanation_positions, selected_positions = _select_explanation_rows(
        X_test_preprocessed,
        y_test,
        all_test_predictions,
        max_rows=min(explanation_rows, len(X_test)),
        random_state=random_state,
    )
    X_explain = X_test_preprocessed.iloc[explanation_positions]
    shap_explanation = explainer(X_explain)
    values, base_values = _normalize_shap_output(
        shap_explanation,
        row_count=len(X_explain),
        feature_count=len(feature_names),
    )
    subset_probabilities = all_test_probabilities[explanation_positions]
    reconstructed = base_values + values.sum(axis=1)
    if not np.allclose(
        reconstructed, subset_probabilities, rtol=1e-4, atol=2e-5
    ):
        maximum_error = float(np.max(np.abs(reconstructed - subset_probabilities)))
        raise AssertionError(
            "Class-1 SHAP values do not reconstruct the baseline model's anomaly "
            f"probabilities (maximum error {maximum_error:.8g})."
        )

    mean_absolute_shap = np.abs(values).mean(axis=0)
    mean_signed_shap = values.mean(axis=0)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": mean_absolute_shap,
            "mean_signed_shap": mean_signed_shap,
        }
    ).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)

    row_lookup = {
        int(test_position): local_position
        for local_position, test_position in enumerate(explanation_positions)
    }
    local_explanations = {}
    local_summaries = {}
    for kind, test_position in selected_positions.items():
        local_position = row_lookup[test_position]
        local_explanation, local_summary = _make_local_explanation(
            position=local_position,
            kind=kind,
            feature_values=X_explain,
            shap_values=values,
            base_values=base_values,
            probabilities=subset_probabilities,
            predictions=all_test_predictions[explanation_positions],
            y_test=y_test.iloc[explanation_positions].reset_index(drop=True),
        )
        local_explanations[kind] = local_explanation
        local_summary["test_row_position"] = int(test_position)
        if raw_dataset_path is not None and Path(raw_dataset_path).is_file():
            raw_frame = pd.read_csv(raw_dataset_path)
            if {"segment", "train", *feature_names}.issubset(raw_frame.columns):
                original_test = raw_frame.loc[raw_frame["train"].eq(0)].reset_index(
                    drop=True
                )
                if len(original_test) != len(X_test):
                    raise ValueError(
                        "Original test rows do not align with processed test rows."
                    )
                if not np.allclose(
                    original_test.loc[:, feature_names].to_numpy(dtype=float),
                    X_test.to_numpy(dtype=float),
                ):
                    raise ValueError(
                        "Original test rows differ from Stage 3 data; refusing to "
                        "associate incorrect segment identifiers."
                    )
                local_summary["segment_id"] = int(
                    original_test.loc[test_position, "segment"]
                )
        local_summaries[kind] = local_summary

    figures_path = Path(figures_dir)
    summary_path = Path(summary_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    summary_path.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.plots.bar(shap_explanation, max_display=18, show=False)
    plt.gcf().set_size_inches(10, 8)
    plt.tight_layout()
    plt.savefig(
        figures_path / "shap_feature_importance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(plt.gcf())

    plt.figure()
    shap.plots.beeswarm(shap_explanation, max_display=18, show=False)
    plt.gcf().set_size_inches(11, 8)
    plt.tight_layout()
    plt.savefig(figures_path / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close(plt.gcf())

    top_feature = str(importance.iloc[0]["feature"])
    plt.figure()
    shap.dependence_plot(
        top_feature,
        values,
        X_explain,
        interaction_index=None,
        show=False,
    )
    plt.gcf().set_size_inches(9, 6)
    plt.tight_layout()
    plt.savefig(
        figures_path / "shap_dependence.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(plt.gcf())

    for kind, filename in (
        ("normal", "shap_normal_prediction.png"),
        ("anomalous", "shap_anomalous_prediction.png"),
    ):
        plt.figure()
        shap.plots.waterfall(
            local_explanations[kind],
            max_display=10,
            show=False,
        )
        plt.gcf().set_size_inches(10, 7)
        plt.tight_layout()
        plt.savefig(figures_path / filename, dpi=180, bbox_inches="tight")
        plt.close(plt.gcf())

    importance_path = summary_path / "shap_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    top_features = importance.head(5).to_dict(orient="records")
    summary = {
        "model": "Baseline XGBoost",
        "artifact": BASELINE_MODEL_FILENAME,
        "output_explained": "P(anomaly=1)",
        "shap_method": "TreeExplainer with interventional feature perturbation",
        "training_background_rows": int(len(background)),
        "test_rows_explained": int(len(X_explain)),
        "random_state": random_state,
        "feature_names": list(feature_names),
        "top_features": top_features,
        "selected_predictions": local_summaries,
        "interpretation": {
            "positive_shap": "raises the model's predicted probability of anomaly",
            "negative_shap": "lowers the model's predicted probability of anomaly",
            "limitation": (
                "SHAP describes this model's behavior, not physical causation or "
                "proof of a satellite fault."
            ),
        },
        "additivity_max_absolute_error": float(
            np.max(np.abs(reconstructed - subset_probabilities))
        ),
    }
    summary_json_path = summary_path / "shap_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    required_figures = [
        "shap_feature_importance.png",
        "shap_summary.png",
        "shap_dependence.png",
        "shap_normal_prediction.png",
        "shap_anomalous_prediction.png",
    ]
    for filename in required_figures:
        path = figures_path / filename
        _require_file(path)
        if path.stat().st_size == 0:
            raise OSError(f"SHAP plot is empty: {path}")

    return {
        "importance": importance,
        "top_features": top_features,
        "selected_predictions": local_summaries,
        "summary": summary,
        "figures": {name: figures_path / name for name in required_figures},
        "summary_files": {
            "importance_csv": importance_path,
            "summary_json": summary_json_path,
        },
        "explanation_rows": int(len(X_explain)),
        "background_rows": int(len(background)),
        "all_test_predictions": all_test_predictions,
        "all_test_probabilities": all_test_probabilities,
    }

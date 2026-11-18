"""Evaluate saved Stage 4 model pipelines on the official held-out test split."""

import json
from pathlib import Path
from typing import Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data_loader import ENGINEERED_FEATURE_COLUMNS, TARGET_COLUMN


MODEL_ARTIFACTS = {
    "Decision Tree": "decision_tree_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "XGBoost baseline": "xgboost_baseline_model.joblib",
    "Tuned XGBoost": "xgboost_model.joblib",
}
TEST_FILENAME = "test_processed.csv"
FEATURE_INFO_FILENAME = "feature_names.json"
MODEL_METADATA_FILENAME = "model_metadata.json"


def _read_test_data(
    processed_dir: Union[str, Path],
    feature_names_path: Union[str, Path],
) -> tuple[pd.DataFrame, pd.Series]:
    """Load Stage 3's official test file and validate its exact feature contract."""
    test_path = Path(processed_dir) / TEST_FILENAME
    metadata_path = Path(feature_names_path)
    missing = [path for path in (test_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required evaluation input is missing: "
            + ", ".join(str(path) for path in missing)
        )

    feature_info = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = feature_info.get("features")
    if feature_names != list(ENGINEERED_FEATURE_COLUMNS):
        raise ValueError(
            "Feature information does not match the Stage 3 benchmark feature list."
        )
    if feature_info.get("target") != TARGET_COLUMN:
        raise ValueError(
            f"Feature information must identify {TARGET_COLUMN!r} as the target."
        )
    excluded = set(feature_info.get("excluded_inputs", []))
    if {"anomaly", "train"}.difference(excluded):
        raise ValueError("Feature information does not explicitly exclude target and split.")

    frame = pd.read_csv(test_path)
    expected_columns = [*ENGINEERED_FEATURE_COLUMNS, TARGET_COLUMN]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"{test_path.name} must contain exactly the Stage 3 feature columns and "
            f"{TARGET_COLUMN!r}. Actual columns: {frame.columns.tolist()}"
        )
    X_test = frame.loc[:, list(ENGINEERED_FEATURE_COLUMNS)]
    y_test = frame[TARGET_COLUMN].astype(int)
    if frame.isna().any().any():
        raise ValueError(f"{test_path} contains missing values.")
    if set(y_test.unique()) != {0, 1}:
        raise ValueError("The official test target must contain both labels 0 and 1.")
    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("The official test features contain non-finite values.")
    return X_test, y_test


def _load_saved_models(
    model_dir: Union[str, Path],
    metadata_path: Union[str, Path],
    feature_names: tuple[str, ...],
) -> dict:
    """Load all required Stage 4 model artifacts or report every missing file."""
    directory = Path(model_dir)
    metadata_file = Path(metadata_path)
    required_paths = [directory / filename for filename in MODEL_ARTIFACTS.values()]
    missing = [path for path in [metadata_file, *required_paths] if not path.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Evaluation cannot proceed because required saved model artifacts are "
            f"missing:\n{details}\nRun the Stage 4 training workflow to create them."
        )

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    saved_artifacts = metadata.get("artifacts", {}).get("models", {})
    models = {}
    for model_name, filename in MODEL_ARTIFACTS.items():
        declared_filename = saved_artifacts.get(
            {
                "Decision Tree": "Decision Tree baseline",
                "Random Forest": "Random Forest baseline",
                "XGBoost baseline": "XGBoost baseline",
                "Tuned XGBoost": "Tuned XGBoost",
            }[model_name]
        )
        if declared_filename != filename:
            raise ValueError(
                f"Stage 4 metadata does not declare the expected {model_name} artifact "
                f"({filename})."
            )
        model = joblib.load(directory / filename)
        if not hasattr(model, "named_steps") or "preprocessing" not in model.named_steps:
            raise TypeError(
                f"{filename} is not the expected saved inference pipeline with "
                "the Stage 3 preprocessing step."
            )
        if "classifier" not in model.named_steps:
            raise TypeError(f"{filename} has no classifier step.")
        fitted_preprocessor = model.named_steps["preprocessing"]
        fitted_features = tuple(
            getattr(fitted_preprocessor, "feature_names_in_", ())
        )
        if fitted_features != feature_names:
            raise ValueError(
                f"{filename} was fitted with unexpected feature names: {fitted_features}"
            )
        model_features = tuple(
            getattr(model.named_steps["classifier"], "feature_names_in_", ())
        )
        if model_features and model_features != feature_names:
            raise ValueError(
                f"{filename} classifier feature names do not match Stage 3."
            )
        models[model_name] = model
    return models


def _load_test_segment_ids(
    raw_dataset_path: Optional[Union[str, Path]],
    y_test: pd.Series,
    X_test: pd.DataFrame,
) -> Optional[list[int]]:
    """Map processed test rows to original segment identifiers when available."""
    if raw_dataset_path is None:
        return None
    source_path = Path(raw_dataset_path)
    if not source_path.is_file():
        return None
    original = pd.read_csv(source_path)
    required = {
        *ENGINEERED_FEATURE_COLUMNS,
        "segment",
        "anomaly",
        "train",
    }
    if not required.issubset(original.columns):
        return None
    original_test = original.loc[original["train"].eq(0)].reset_index(drop=True)
    if len(original_test) != len(y_test):
        raise ValueError(
            "Raw dataset test row count does not match the Stage 3 processed test file."
        )
    original_features = original_test.loc[:, list(ENGINEERED_FEATURE_COLUMNS)]
    if not np.allclose(original_features.to_numpy(dtype=float), X_test.to_numpy(dtype=float)):
        raise ValueError(
            "Raw dataset features do not align with the Stage 3 test rows; "
            "refusing to report potentially incorrect segment identifiers."
        )
    if not np.array_equal(
        original_test["anomaly"].to_numpy(dtype=int), y_test.to_numpy(dtype=int)
    ):
        raise ValueError(
            "Raw dataset labels do not align with the Stage 3 test rows."
        )
    return original_test["segment"].astype(int).tolist()


def _save_confusion_matrix(
    results: dict,
    output_path: Path,
) -> None:
    """Save comparable count-based confusion matrices for every model."""
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (model_name, result) in zip(axes.flat, results.items()):
        sns.heatmap(
            result["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            xticklabels=["Normal (0)", "Anomalous (1)"],
            yticklabels=["Normal (0)", "Anomalous (1)"],
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
    fig.suptitle("Held-out test confusion matrices", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_roc_curve(results: dict, y_test: pd.Series, output_path: Path) -> None:
    """Plot all model ROC curves with a random-classification reference line."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for model_name, result in results.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, result["probabilities"]
        )
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{model_name} (AUC={result['metrics']['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set(
        title="ROC curves on the official held-out test set",
        xlabel="False positive rate",
        ylabel="True positive rate (anomaly recall)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_precision_recall_curve(
    results: dict,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Plot all model precision-recall curves and the test-set prevalence."""
    prevalence = float(y_test.mean())
    fig, ax = plt.subplots(figsize=(8, 7))
    for model_name, result in results.items():
        precision, recall, _ = precision_recall_curve(
            y_test, result["probabilities"]
        )
        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model_name} (PR-AUC={result['metrics']['pr_auc']:.3f})",
        )
    ax.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        label=f"Test anomaly prevalence ({prevalence:.3f})",
    )
    ax.set(
        title="Precision-recall curves on the official held-out test set",
        xlabel="Recall (anomaly detection rate)",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_saved_models(
    model_dir: Union[str, Path],
    processed_dir: Union[str, Path],
    report_dir: Union[str, Path],
    figures_dir: Union[str, Path],
    feature_names_path: Optional[Union[str, Path]] = None,
    model_metadata_path: Optional[Union[str, Path]] = None,
    raw_dataset_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Evaluate every saved Stage 4 model on the official held-out test records."""
    model_directory = Path(model_dir)
    feature_info_path = (
        Path(feature_names_path)
        if feature_names_path is not None
        else model_directory / FEATURE_INFO_FILENAME
    )
    stage4_metadata_path = (
        Path(model_metadata_path)
        if model_metadata_path is not None
        else model_directory / MODEL_METADATA_FILENAME
    )
    X_test, y_test = _read_test_data(processed_dir, feature_info_path)
    models = _load_saved_models(
        model_directory,
        stage4_metadata_path,
        ENGINEERED_FEATURE_COLUMNS,
    )
    segment_ids = _load_test_segment_ids(raw_dataset_path, y_test, X_test)

    results = {}
    prediction_rows = []
    for model_name, model in models.items():
        predictions = np.asarray(model.predict(X_test), dtype=int)
        classifier = model.named_steps["classifier"]
        class_labels = np.asarray(classifier.classes_)
        if not np.array_equal(class_labels, np.array([0, 1])):
            raise ValueError(
                f"{model_name} has unsupported class labels: {class_labels.tolist()}"
            )
        probabilities = np.asarray(model.predict_proba(X_test))[:, 1]
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{model_name} produced non-finite probabilities.")

        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_test, probabilities
        )
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_anomaly": float(
                precision_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "recall_anomaly": float(
                recall_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "f1_anomaly": float(
                f1_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(auc(recall_curve, precision_curve)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
        }
        report_dict = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal (0)", "Anomalous (1)"],
            output_dict=True,
            zero_division=0,
        )
        false_positive_ids = (
            [
                segment_ids[index]
                for index, (actual, predicted) in enumerate(
                    zip(y_test.to_numpy(), predictions)
                )
                if actual == 0 and predicted == 1
            ]
            if segment_ids is not None
            else None
        )
        false_negative_ids = (
            [
                segment_ids[index]
                for index, (actual, predicted) in enumerate(
                    zip(y_test.to_numpy(), predictions)
                )
                if actual == 1 and predicted == 0
            ]
            if segment_ids is not None
            else None
        )
        results[model_name] = {
            "metrics": metrics,
            "confusion_matrix": matrix.tolist(),
            "confusion_counts": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
            "classification_report": report_dict,
            "false_positive_segment_ids": false_positive_ids,
            "false_negative_segment_ids": false_negative_ids,
            "probabilities": probabilities,
            "predictions": predictions,
        }

        for index, (actual, predicted, probability) in enumerate(
            zip(y_test.to_numpy(), predictions, probabilities)
        ):
            prediction_rows.append(
                {
                    "model": model_name,
                    "segment": segment_ids[index] if segment_ids is not None else None,
                    "actual": int(actual),
                    "predicted": int(predicted),
                    "anomaly_probability": float(probability),
                    "outcome": (
                        "true_positive"
                        if actual == 1 and predicted == 1
                        else "false_negative"
                        if actual == 1
                        else "false_positive"
                        if predicted == 1
                        else "true_negative"
                    ),
                }
            )

    report_directory = Path(report_dir)
    figure_directory = Path(figures_dir)
    report_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for model_name, result in results.items():
        comparison_rows.append(
            {
                "model": model_name,
                **result["metrics"],
                **result["confusion_counts"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    # Recall is the primary ranking criterion for anomaly detection; F1 breaks ties.
    comparison = comparison.sort_values(
        ["recall_anomaly", "f1_anomaly", "pr_auc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    comparison.to_csv(report_directory / "model_performance.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        report_directory / "evaluation_predictions.csv", index=False
    )

    report_lines = [
        "ORBITGUARD Stage 5 — Model evaluation",
        "Evaluation data: official held-out test split only",
        f"Test records: {len(y_test)}",
        f"Class counts: {y_test.value_counts().sort_index().to_dict()} (0=Normal, 1=Anomalous)",
        "PR-AUC is the trapezoidal area under the precision-recall curve; "
        "average precision is reported separately.",
        "Models are ranked by anomaly recall, then anomaly F1, then PR-AUC.",
        "",
    ]
    for model_name, result in results.items():
        tn, fp, fn, tp = (
            result["confusion_counts"][key]
            for key in (
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
            )
        )
        report_lines.extend(
            [
                f"=== {model_name} ===",
                f"Metrics: {json.dumps(result['metrics'], sort_keys=True)}",
                f"Confusion counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
                "Classification report:",
                classification_report(
                    y_test,
                    result["predictions"],
                    labels=[0, 1],
                    target_names=["Normal (0)", "Anomalous (1)"],
                    zero_division=0,
                ).rstrip(),
                "False-positive segment IDs: "
                + (
                    ", ".join(map(str, result["false_positive_segment_ids"]))
                    if result["false_positive_segment_ids"] is not None
                    else "not available (original dataset not supplied)"
                ),
                "False-negative segment IDs: "
                + (
                    ", ".join(map(str, result["false_negative_segment_ids"]))
                    if result["false_negative_segment_ids"] is not None
                    else "not available (original dataset not supplied)"
                ),
                "",
            ]
        )
    (report_directory / "classification_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    best_model = comparison.iloc[0]["model"]
    dashboard_data = {
        "project": "ORBITGUARD",
        "stage": 5,
        "evaluation": {
            "split": "official held-out test",
            "records": int(len(y_test)),
            "class_mapping": {"0": "Normal", "1": "Anomalous"},
            "class_counts": {
                str(key): int(value)
                for key, value in y_test.value_counts().sort_index().items()
            },
            "threshold": 0.5,
            "pr_auc_definition": "trapezoidal area under precision_recall_curve",
            "ranking_rule": [
                "recall_anomaly descending",
                "f1_anomaly descending",
                "pr_auc descending",
            ],
            "best_model": best_model,
        },
        "models": {
            name: {
                **result["metrics"],
                "confusion_matrix": result["confusion_matrix"],
                "confusion_counts": result["confusion_counts"],
                "classification_report": result["classification_report"],
                "false_positive_segment_ids": result["false_positive_segment_ids"],
                "false_negative_segment_ids": result["false_negative_segment_ids"],
            }
            for name, result in results.items()
        },
    }
    (report_directory / "evaluation_results.json").write_text(
        json.dumps(dashboard_data, indent=2), encoding="utf-8"
    )

    _save_confusion_matrix(
        results, figure_directory / "confusion_matrix.png"
    )
    _save_roc_curve(results, y_test, figure_directory / "roc_curve.png")
    _save_precision_recall_curve(
        results, y_test, figure_directory / "precision_recall_curve.png"
    )

    return {
        "comparison": comparison,
        "results": results,
        "dashboard_data": dashboard_data,
        "best_model": best_model,
        "test_class_counts": y_test.value_counts().sort_index().to_dict(),
        "report_paths": {
            "performance_csv": report_directory / "model_performance.csv",
            "classification_report": report_directory / "classification_report.txt",
            "dashboard_json": report_directory / "evaluation_results.json",
            "predictions_csv": report_directory / "evaluation_predictions.csv",
            "confusion_matrix": figure_directory / "confusion_matrix.png",
            "roc_curve": figure_directory / "roc_curve.png",
            "precision_recall_curve": figure_directory / "precision_recall_curve.png",
        },
    }


def main() -> None:
    """Evaluate saved models from the project root without fitting them."""
    project_root = Path(__file__).resolve().parents[1]
    result = evaluate_saved_models(
        model_dir=project_root / "models",
        processed_dir=project_root / "data" / "processed",
        report_dir=project_root / "reports" / "evaluation",
        figures_dir=project_root / "reports" / "figures",
        raw_dataset_path=project_root / "data" / "raw" / "dataset.csv",
    )
    print(result["comparison"].to_string(index=False))
    print(f"\nBest model by anomaly recall, then F1 and PR-AUC: {result['best_model']}")
    for path in result["report_paths"].values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

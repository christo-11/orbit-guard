"""Read-only loaders for the existing ORBITGUARD dashboard artifacts."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ARTIFACTS = (
    Path("data/raw/dataset.csv"),
    Path("data/raw/segments.csv"),
    Path("models/xgboost_baseline_model.joblib"),
    Path("models/feature_names.json"),
    Path("reports/evaluation/model_performance.csv"),
    Path("reports/evaluation/evaluation_predictions.csv"),
    Path("reports/evaluation/evaluation_results.json"),
    Path("reports/evaluation/shap_feature_importance.csv"),
    Path("reports/evaluation/shap_summary.json"),
)

FEATURE_COLUMNS = (
    "duration",
    "len",
    "mean",
    "var",
    "std",
    "kurtosis",
    "skew",
    "n_peaks",
    "smooth10_n_peaks",
    "smooth20_n_peaks",
    "diff_peaks",
    "diff2_peaks",
    "diff_var",
    "diff2_var",
    "gaps_squared",
    "len_weighted",
    "var_div_duration",
    "var_div_len",
)

MODEL_NAMES = (
    "Decision Tree",
    "Random Forest",
    "XGBoost baseline",
    "Tuned XGBoost",
)


class DashboardDataError(RuntimeError):
    """Raised when a required dashboard artifact is absent or inconsistent."""


def find_missing_artifacts(root: Path = PROJECT_ROOT) -> list[Path]:
    """Return all required files that are missing, for one clear UI message."""
    return [root / relative_path for relative_path in REQUIRED_ARTIFACTS if not (root / relative_path).is_file()]


def load_dashboard_tables(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load real project tables and verify their expected schemas."""
    missing = find_missing_artifacts(root)
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise DashboardDataError(
            "The dashboard needs Stage 1–6 artifacts that are missing:\n"
            f"{missing_list}\n"
            "Complete the required project stages and try again."
        )

    feature_info = json.loads(
        (root / "models" / "feature_names.json").read_text(encoding="utf-8")
    )
    feature_names = feature_info.get("features")
    if feature_names != list(FEATURE_COLUMNS):
        raise DashboardDataError(
            "models/feature_names.json does not match the verified Stage 3 "
            "feature list. Refusing to score incompatible data."
        )
    if feature_info.get("target") != "anomaly":
        raise DashboardDataError("The saved feature contract has an unexpected target.")

    records = pd.read_csv(root / "data" / "raw" / "dataset.csv")
    required_record_columns = {
        *FEATURE_COLUMNS,
        "segment",
        "anomaly",
        "train",
        "channel",
        "sampling",
    }
    if not required_record_columns.issubset(records.columns):
        raise DashboardDataError("The original feature dataset has an unexpected schema.")
    if records["segment"].duplicated().any():
        raise DashboardDataError("Segment identifiers must be unique for dashboard selection.")

    raw_telemetry = pd.read_csv(root / "data" / "raw" / "segments.csv")
    required_telemetry_columns = {
        "segment",
        "timestamp",
        "value",
        "channel",
        "anomaly",
    }
    if not required_telemetry_columns.issubset(raw_telemetry.columns):
        raise DashboardDataError("The raw telemetry file has an unexpected schema.")

    performance = pd.read_csv(root / "reports" / "evaluation" / "model_performance.csv")
    if set(performance["model"]) != set(MODEL_NAMES):
        raise DashboardDataError(
            "Model performance report does not contain all four expected models."
        )
    predictions = pd.read_csv(
        root / "reports" / "evaluation" / "evaluation_predictions.csv"
    )
    if not {
        "model",
        "segment",
        "actual",
        "predicted",
        "anomaly_probability",
        "outcome",
    }.issubset(predictions.columns):
        raise DashboardDataError("The Stage 5 prediction report has an unexpected schema.")
    if set(predictions["model"].unique()) != set(MODEL_NAMES):
        raise DashboardDataError(
            "Saved prediction records do not include all four evaluated models."
        )

    evaluation = json.loads(
        (root / "reports" / "evaluation" / "evaluation_results.json").read_text(
            encoding="utf-8"
        )
    )
    shap_importance = pd.read_csv(
        root / "reports" / "evaluation" / "shap_feature_importance.csv"
    )
    if not {"feature", "mean_absolute_shap", "mean_signed_shap"}.issubset(
        shap_importance.columns
    ):
        raise DashboardDataError("The Stage 6 SHAP importance report has an unexpected schema.")
    shap_summary = json.loads(
        (root / "reports" / "evaluation" / "shap_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if shap_summary.get("feature_names") != list(FEATURE_COLUMNS):
        raise DashboardDataError("The SHAP summary feature names do not match Stage 3.")

    return {
        "records": records,
        "raw_telemetry": raw_telemetry,
        "performance": performance,
        "predictions": predictions,
        "evaluation": evaluation,
        "shap_importance": shap_importance,
        "shap_summary": shap_summary,
        "feature_names": feature_names,
        "figure_dir": root / "reports" / "figures",
        "root": root,
    }


def load_baseline_model(root: Path = PROJECT_ROOT) -> Any:
    """Load and validate the saved baseline XGBoost inference pipeline."""
    model_path = root / "models" / "xgboost_baseline_model.joblib"
    feature_info_path = root / "models" / "feature_names.json"
    missing = [path for path in (model_path, feature_info_path) if not path.is_file()]
    if missing:
        raise DashboardDataError(
            "Required prediction artifact is missing:\n"
            + "\n".join(f"- {path}" for path in missing)
        )

    feature_info = json.loads(feature_info_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "named_steps") or not {
        "preprocessing",
        "classifier",
    }.issubset(pipeline.named_steps):
        raise DashboardDataError(
            "Saved baseline XGBoost model must include the Stage 3 preprocessing step."
        )
    if tuple(pipeline.named_steps["preprocessing"].feature_names_in_) != tuple(
        feature_info["features"]
    ):
        raise DashboardDataError(
            "Saved model preprocessing features do not match feature_names.json."
        )
    if tuple(pipeline.named_steps["classifier"].feature_names_in_) != tuple(
        feature_info["features"]
    ):
        raise DashboardDataError(
            "Saved XGBoost classifier features do not match feature_names.json."
        )
    return pipeline


def predict_segment(pipeline: Any, record: pd.Series, feature_names: list[str]) -> dict:
    """Score one actual dataset segment using the saved inference pipeline."""
    features = pd.DataFrame(
        [{name: record[name] for name in feature_names}],
        columns=feature_names,
    )
    if features.columns.tolist() != feature_names:
        raise DashboardDataError("Selected segment features do not match the model.")
    predicted_class = int(pipeline.predict(features)[0])
    probabilities = pipeline.predict_proba(features)[0]
    classifier_classes = list(pipeline.named_steps["classifier"].classes_)
    anomaly_probability = float(probabilities[classifier_classes.index(1)])
    if not np.isfinite(anomaly_probability):
        raise DashboardDataError("The saved model returned an invalid prediction probability.")
    return {
        "predicted_class": predicted_class,
        "predicted_label": "Anomalous" if predicted_class == 1 else "Normal",
        "anomaly_probability": anomaly_probability,
        "normal_probability": 1.0 - anomaly_probability,
    }


def explain_selected_segment(
    pipeline: Any,
    record: pd.Series,
    feature_names: list[str],
    train_path: Path,
    background_rows: int = 80,
    random_state: int = 42,
) -> dict:
    """Compute real on-demand local SHAP contributions for one selected segment."""
    import shap

    if not train_path.is_file():
        raise DashboardDataError(
            f"Stage 3 training data is needed for SHAP background: {train_path}"
        )
    train = pd.read_csv(train_path, usecols=feature_names)
    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]
    transformed_train = pd.DataFrame(
        preprocessor.transform(train),
        columns=feature_names,
    )
    background = transformed_train.sample(
        n=min(background_rows, len(transformed_train)),
        random_state=random_state,
    )
    one_record = pd.DataFrame(
        [{name: record[name] for name in feature_names}],
        columns=feature_names,
    )
    transformed_record = pd.DataFrame(
        preprocessor.transform(one_record),
        columns=feature_names,
    )
    explainer = shap.TreeExplainer(
        classifier,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    explanation = explainer(transformed_record)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        if values.shape == (1, len(feature_names), 2):
            values = values[0, :, 1]
        elif values.shape == (2, 1, len(feature_names)):
            values = values[1, 0, :]
        else:
            raise DashboardDataError(f"Unexpected binary SHAP output shape: {values.shape}")
    elif values.ndim == 2:
        values = values[0]
    if values.shape != (len(feature_names),):
        raise DashboardDataError(f"Unexpected local SHAP shape: {values.shape}")
    base_values = np.asarray(explanation.base_values).reshape(-1)
    if base_values.size == 2:
        base_value = float(base_values[1])
    elif base_values.size == 1:
        base_value = float(base_values[0])
    else:
        raise DashboardDataError(
            f"Unexpected local SHAP base value shape: {base_values.shape}"
        )
    probability = float(pipeline.predict_proba(one_record)[0, 1])
    reconstructed = float(base_value + values.sum())
    if not np.isclose(reconstructed, probability, rtol=1e-4, atol=2e-5):
        raise DashboardDataError(
            "Local SHAP values did not reconstruct this model's anomaly probability."
        )
    return {
        "contributions": pd.DataFrame(
            {
                "feature": feature_names,
                "feature_value": transformed_record.iloc[0].to_numpy(dtype=float),
                "shap_value": values,
            }
        ).assign(abs_shap=lambda frame: frame["shap_value"].abs())
        .sort_values("abs_shap", ascending=False),
        "base_probability": base_value,
        "anomaly_probability": probability,
        "reconstructed_probability": reconstructed,
    }

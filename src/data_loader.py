"""Load the original OPSSAT-AD CSV files without changing their labels."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURE_DATASET_FILENAME = "dataset.csv"
RAW_SEGMENTS_FILENAME = "segments.csv"

TARGET_COLUMN = "anomaly"
SPLIT_COLUMN = "train"

# This allowlist contains the 18 benchmark features, not IDs, labels, or metadata.
ENGINEERED_FEATURE_COLUMNS = (
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


def _read_csv(filename: str, data_dir: Optional[Union[str, Path]]) -> pd.DataFrame:
    """Read one downloaded CSV and explain how to resolve a missing-file error."""
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    csv_path = directory / filename
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. "
            "Run notebooks/01_data_collection.ipynb to download and verify the "
            "original Zenodo files."
        )
    return pd.read_csv(csv_path)


def load_feature_dataset(
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load dataset.csv, preserving its original target and train/test columns."""
    return _read_csv(FEATURE_DATASET_FILENAME, data_dir)


def load_raw_segments(
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load segments.csv, preserving the original telemetry and label columns."""
    return _read_csv(RAW_SEGMENTS_FILENAME, data_dir)


def select_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the documented engineered features, excluding metadata/labels."""
    missing_columns = [
        column for column in ENGINEERED_FEATURE_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "The feature table is missing expected benchmark feature columns: "
            + ", ".join(missing_columns)
        )
    return frame.loc[:, list(ENGINEERED_FEATURE_COLUMNS)].copy()

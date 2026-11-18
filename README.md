# ORBITGUARD

**Explainable Satellite Telemetry Anomaly Intelligence** — dataset collection and exploratory analysis.

This project starts with OPSSAT-AD, an ESA OPS-SAT telemetry anomaly benchmark. Stage 1 downloads and verifies the original files, documents their schemas, and preserves the benchmark labels and split. It does not perform exploratory analysis or train a model.

## Dataset source

- Zenodo record: [OPSSAT-AD - anomaly detection dataset for satellite telemetry](https://zenodo.org/records/12588359)
- Dataset DOI: [10.5281/zenodo.12588359](https://doi.org/10.5281/zenodo.12588359)
- Reference repository: [kplabs-pl/OPS-SAT-AD](https://github.com/kplabs-pl/OPS-SAT-AD)
- Benchmark paper: [The OPS-SAT benchmark for detecting anomalies in satellite telemetry](https://arxiv.org/abs/2407.04730)

The Zenodo record lists two downloadable CSV files. Both files were downloaded from the record's file API and their MD5 checksums matched the published metadata:

| File | Downloaded size (bytes) | Published MD5 |
|---|---:|---|
| `dataset.csv` | 507,568 | `c12fb83863afa4463dec07303a20bb14` |
| `segments.csv` | 17,970,946 | `1f4b6ffd0359efac96a6698ff557e0f9` |

Original files belong in `data/raw/`. They are intentionally excluded from Git; use the notebook to download them again. `data/processed/` is reserved for later stages.

## Verified dataset contents

The following details come from reading both downloaded CSVs. The schema and sample records are also printed when the collection notebook runs.

| File | Shape | Contents |
|---|---:|---|
| `dataset.csv` | 2,123 rows × 23 columns | One row per telemetry segment, with the benchmark's engineered features, target, split, and metadata |
| `segments.csv` | 303,493 rows × 8 columns | Timestamped telemetry values grouped by segment, with labels and metadata repeated on the samples |

`dataset.csv` columns:

```text
segment, anomaly, train, channel, sampling, duration, len, mean, var, std,
kurtosis, skew, n_peaks, smooth10_n_peaks, smooth20_n_peaks, diff_peaks,
diff2_peaks, diff_var, diff2_var, gaps_squared, len_weighted,
var_div_duration, var_div_len
```

In the inspected environment (pandas 3.0.5), `segment`, `anomaly`, `train`, `sampling`, `duration`, `len`, `n_peaks`, `smooth10_n_peaks`, `smooth20_n_peaks`, `diff_peaks`, `diff2_peaks`, `gaps_squared`, and `len_weighted` are `int64`; the other numeric feature columns are `float64`; and `channel` is a string. Pandas may represent strings as `object` with other pandas versions.

`segments.csv` columns:

```text
channel, timestamp, value, label, sampling, anomaly, segment, train
```

`value` is `float64`; `sampling`, `anomaly`, `segment`, and `train` are `int64`; and `channel`, `timestamp`, and `label` are strings in the inspected environment. All columns in both files had zero missing values in this download.

## Target, labels, and official split

- The segment-level target is `anomaly` in `dataset.csv`; its observed values are `0` and `1`.
- The raw telemetry file also has a `label` column. Its observed values are `a2`, `a3`, `a4`, and `anomaly`. In this data, `anomaly=1` corresponds to `label="anomaly"`; `anomaly=0` corresponds to labels `a2`, `a3`, or `a4`. The original label columns are not rewritten.
- At segment level, `anomaly=0` occurs 1,689 times and `anomaly=1` occurs 434 times. In `segments.csv`, the sample-level `label` counts are `a2`: 75,272; `a3`: 106,930; `a4`: 21,027; and `anomaly`: 100,264. These sample counts are not segment counts.
- The benchmark paper describes a stratified random train/test split and instructs users to use the `train` attribute in `dataset.csv`. Observed values are `train=1` for training and `train=0` for test:

| `train` value | Split | Segments | `anomaly=0` | `anomaly=1` |
|---:|---|---:|---:|---:|
| 1 | Train | 1,594 | 1,273 | 321 |
| 0 | Test | 529 | 416 | 113 |

Use this supplied split for benchmark-compatible work rather than creating a new random split.

## Raw telemetry and engineered features

`segments.csv` contains the sample-level telemetry (`timestamp` and `value`) grouped by `segment`. `dataset.csv` contains the benchmark's 18 handcrafted segment-level features, plus the target, split, and metadata. The 18-feature allowlist used by `src/data_loader.py` is:

```text
duration, len, mean, var, std, kurtosis, skew, n_peaks,
smooth10_n_peaks, smooth20_n_peaks, diff_peaks, diff2_peaks,
diff_var, diff2_var, gaps_squared, len_weighted,
var_div_duration, var_div_len
```

The loader provides this allowlist separately; it does not use `segment` (identifier), `channel` or `sampling` (metadata), `anomaly` (target), or `train` (split) as model features.

## Setup and download

Python 3.9 or newer is supported. From the project root:

```bash
python -m pip install -r requirements.txt
```

Run the notebook locally:

```bash
jupyter notebook notebooks/01_data_collection.ipynb
```

Or open the notebook in Google Colab. Run its cells from top to bottom. It downloads both CSVs from Zenodo, verifies their published MD5 checksums, and prints each file's shape, columns, inferred data types, and sample records. It also prints the observed labels and benchmark split counts. A checksum mismatch or download failure is reported rather than silently accepted.

To load the downloaded data from Python:

```python
from src.data_loader import load_feature_dataset, load_raw_segments

features_and_labels = load_feature_dataset()
raw_telemetry = load_raw_segments()
```

If the files have not been downloaded yet, run the notebook first. The loader does not download data or change labels.

## Stage 2: Exploratory Data Analysis

Open and run [`notebooks/02_exploratory_data_analysis.ipynb`](notebooks/02_exploratory_data_analysis.ipynb) from top to bottom after Stage 1. The notebook inspects dataset dimensions and dtypes, missing and duplicate records, descriptive statistics, target and official split distributions, all 18 engineered feature histograms, feature correlations, class-wise boxplots and 1.5×IQR outlier counts, channel-level anomaly counts/rates, and example raw telemetry traces. Every visualization includes a plain-language reading guide.

The notebook reads the original files without changing them and does not perform preprocessing, feature selection, or model training. Figures are saved under `reports/figures/`. To reproduce them in VS Code or Colab, install `requirements.txt` and run the notebook. The figures and data files are local outputs; large original data files are excluded from Git.

## Stage 3: Data preprocessing

Run [`notebooks/03_data_preprocessing.ipynb`](notebooks/03_data_preprocessing.ipynb) after Stage 1. The reusable logic is in [`src/preprocessing.py`](src/preprocessing.py). It keeps the official `train=1`/`train=0` split and the original `anomaly` labels, selects only the 18 documented numeric benchmark features, and excludes `segment`, `channel`, `sampling`, `anomaly`, and `train` from the feature matrices.

The current downloaded table has no missing values or duplicate rows. Median imputation is included as a safeguard and is fitted only on training features before transforming test features. Outliers are reported with 1.5×IQR fences on training data and retained; scaling and resampling are not applied. The notebook writes `train_processed.csv`, `test_processed.csv`, `preprocessing_metadata.json`, and the fitted `preprocessing_pipeline.joblib` to `data/processed/`, which is excluded from Git with the original data. It reports feature/target shapes, class counts, remaining missing values, and leakage checks. No model is trained in Stage 3.

## Stage 4: Model development

Run [`notebooks/04_xgboost_model.ipynb`](notebooks/04_xgboost_model.ipynb) after Stage 3. It compares Decision Tree, Random Forest, XGBoost baseline, and tuned XGBoost models. Cross-validation and the compact randomized parameter search use only the official training partition; the official test partition is held out until final metric reporting. Average precision is used for model selection because anomalies are the minority class. The notebook does not resample the data or claim a winning model without considering measured results.

The reusable training workflow is [`src/train.py`](src/train.py). It saves loadable inference pipelines for all four models, including tuned XGBoost and the Stage 3 preprocessor, plus feature information, comparison metrics, and model metadata to `models/`. Model files are local artifacts and are excluded from Git.

## Stage 5: Model evaluation

Run [`notebooks/05_model_evaluation.ipynb`](notebooks/05_model_evaluation.ipynb) after Stage 4. The evaluator loads all four saved inference pipelines, including their Stage 3 preprocessing steps, and reports performance on the official held-out test set without fitting or tuning. It reports accuracy, anomaly precision/recall/F1, ROC-AUC, trapezoidal PR-AUC, average precision, confusion counts, classification reports, and per-segment false-positive/false-negative details. Results are ranked by anomaly recall, then F1, then PR-AUC; this makes the comparison rule explicit rather than implying one model is best under every metric.

Outputs are saved under `reports/evaluation/` (`model_performance.csv`, `classification_report.txt`, dashboard-ready `evaluation_results.json`, and `evaluation_predictions.csv`) and `reports/figures/` (confusion matrices, ROC, and precision-recall curves). All saved models are required; missing artifacts are reported explicitly and are not silently retrained.

## Stage 6: SHAP explainability

Install the requirements including SHAP, then run [`notebooks/06_shap_explainability.ipynb`](notebooks/06_shap_explainability.ipynb). The workflow explains the saved baseline XGBoost model using `TreeExplainer` with a deterministic training background and a representative subset of held-out test data. It checks the feature contract, preprocessing equivalence, unchanged predictions, and SHAP additivity for the class-1 anomaly probability.

The notebook saves global importance, beeswarm, dependence, and selected normal/anomalous waterfall plots to `reports/figures/`. A ranked feature contribution table and reusable JSON summary are written to `reports/evaluation/`. Positive SHAP contributions increase the model's anomaly score relative to its baseline; negative contributions decrease it. These explain model behavior only, not physical causation or proof of a satellite fault.

## Stage 7.1: 3D aerospace AI dashboard redesign

Install the requirements and launch the Streamlit application from the project root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

## React dashboard starter

The separate [`frontend/`](frontend/) project is a React + TypeScript + Tailwind CSS mission-control interface. It includes a dark design system, responsive satellite fleet overview, a Zustand store, and six deterministic demonstration satellites with 24 hourly telemetry values for battery voltage, solar-panel temperature, and gyroscope drift. The mock AI assessments are illustrative and are not connected to the Python model or live spacecraft data.

Run the frontend from the project root:

```bash
cd frontend
npm install
npm run dev
```

Run `npm run build` in `frontend/` to type-check and create the production build.

The dashboard reads the existing Stage 1 telemetry tables, Stage 4 saved baseline XGBoost pipeline, Stage 5 evaluation metrics/predictions, and Stage 6 SHAP reports and figures. It does not train models or invent predictions. The Prediction page scores a selected existing segment with the saved inference pipeline; optional local SHAP contributions are calculated on demand with the Stage 6 SHAP method/background. Predictions and SHAP values describe model behavior and require expert review; they are not satellite fault diagnoses.

The responsive overview pairs a mission-intelligence introduction and dataset summary with an interactive, dotted Three.js globe. The dashboard uses restrained motion, layered glass surfaces, and depth on hover; system reduced-motion preferences are respected. The globe supports pause/resume, reset, drag-to-rotate, and wheel zoom. Its decorative markers and orbital paths are explicitly illustrative and do not represent georeferenced or live telemetry. If the Three.js CDN is unavailable, the visualization falls back to a lightweight 2D canvas globe. Navigation groups model evaluation and saved SHAP explanations under **Model Insights**.

## Project layout

```text
.
├── data/
│   ├── processed/
│   └── raw/
├── dashboard/
│   ├── app.py
│   ├── components.py
│   ├── data_loader.py
│   ├── globe.html
│   ├── globe.py
│   └── styles.css
├── models/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_xgboost_model.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_shap_explainability.ipynb
├── reports/
│   ├── evaluation/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
├── .gitignore
├── README.md
└── requirements.txt
```

Stage 7 reads these saved outputs and provides an interactive local dashboard. It does not modify or retrain any previous-stage artifact.
python -m streamlit run dashboard/app.py
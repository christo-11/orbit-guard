"""ORBITGUARD aerospace AI dashboard."""

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components import callout, hero, section_header, top_navigation
from dashboard.data_loader import (
    explain_selected_segment,
    load_baseline_model,
    load_dashboard_tables,
    predict_segment,
)
from dashboard.globe import render_globe


PAGE_OPTIONS = (
    "Overview",
    "Telemetry",
    "Predictions",
    "Model Insights",
    "About",
)

MODEL_ORDER = (
    "Decision Tree",
    "Random Forest",
    "XGBoost baseline",
    "Tuned XGBoost",
)

PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, 'Segoe UI', sans-serif", "color": "#c3d0df"},
    "margin": {"l": 16, "r": 16, "t": 48, "b": 18},
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
}
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
COLORS = {
    "Normal": "#41d9ec",
    "Anomalous": "#ff7185",
    "Decision Tree": "#91a4ba",
    "Random Forest": "#55e0bc",
    "XGBoost baseline": "#428bff",
    "Tuned XGBoost": "#c08bff",
}

st.set_page_config(
    page_title="ORBITGUARD | Satellite Anomaly Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    css_path = PROJECT_ROOT / "dashboard" / "styles.css"
    if not css_path.is_file():
        st.error(f"Dashboard styling file is missing: {css_path}")
        return
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_dashboard_tables(root_string: str) -> dict:
    return load_dashboard_tables(Path(root_string))


@st.cache_resource(show_spinner="Loading saved baseline XGBoost model…")
def get_model(root_string: str):
    return load_baseline_model(Path(root_string))


@st.cache_data(show_spinner=False)
def get_segment_prediction(root_string: str, segment_id: int) -> dict:
    data = get_dashboard_tables(root_string)
    pipeline = get_model(root_string)
    record = data["records"].loc[data["records"]["segment"].eq(segment_id)].iloc[0]
    return predict_segment(pipeline, record, data["feature_names"])


def metric_value(row: pd.Series, key: str, percent: bool = False) -> str:
    value = float(row[key])
    return f"{value:.1%}" if percent else f"{value:.3f}"


def show_figure(path: Path, caption: str = "") -> None:
    if not path.is_file():
        st.warning(f"Saved figure is not available: {path}")
        return
    st.image(str(path), caption=caption, width="stretch")


def render_overview(data: dict, selected_model: str) -> None:
    records = data["records"]
    performance = data["performance"].set_index("model")
    selected = performance.loc[selected_model]
    total = int(len(records))
    normal = int(records["anomaly"].eq(0).sum())
    anomalous = int(records["anomaly"].eq(1).sum())
    anomaly_share = anomalous / total if total else 0

    intro, visualization = st.columns([1.05, 1.2], gap="large", vertical_alignment="center")
    with intro:
        hero(
            "Satellite Intelligence. Made Explainable.",
            "A clearer view of telemetry anomalies, model performance, and the signals behind every prediction.",
        )
        st.markdown(
            f"""
            <div class="og-hero-tags" aria-label="Dashboard data summary">
              <span><i></i> {total:,} labeled segments</span>
              <span>OPS-SAT · XGBoost · SHAP</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with visualization:
        try:
            render_globe(height=410)
        except FileNotFoundError as error:
            st.warning(f"The 3D visualization could not load: {error}")

    st.markdown(
        '<div class="og-stat-label">OPS-SAT / VERIFIED PROJECT DATA</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    metrics = (
        ("Total telemetry segments", f"{total:,}"),
        ("Normal segments", f"{normal:,}"),
        ("Anomalous segments", f"{anomalous:,}"),
        ("Model accuracy · XGBoost", metric_value(selected, "accuracy", True)),
    )
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.markdown(
                f'<div class="og-stat"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>',
                unsafe_allow_html=True,
            )
    st.caption(
        f"Accuracy comes from the saved held-out test evaluation. "
        f"Dataset anomaly prevalence: {anomaly_share:.1%}. Globe markers and paths are illustrative, not telemetry locations."
    )


def render_explorer(data: dict) -> None:
    records = data["records"]
    hero(
        "Telemetry Explorer",
        "Inspect a labeled segment, compare feature values across classes, and review its raw signal trace.",
        kicker="DATA / SEGMENT INSPECTION",
    )
    section_header(
        "Segment selection",
        "Explore observed telemetry",
        "Segment identifiers and labels are read from the original dataset; model input excludes identifiers and target labels.",
    )
    channel_names = ["All channels"] + sorted(records["channel"].dropna().unique().tolist())
    channel = st.selectbox("Filter by channel", channel_names)
    filtered = records if channel == "All channels" else records.loc[records["channel"].eq(channel)]
    selected_segment = st.selectbox(
        "Telemetry segment",
        options=filtered["segment"].astype(int).tolist(),
        format_func=lambda value: (
            f"SEG-{value:04d} · "
            f"{'Anomalous' if int(filtered.loc[filtered['segment'].eq(value), 'anomaly'].iloc[0]) else 'Normal'}"
        ),
    )
    record = records.loc[records["segment"].eq(selected_segment)].iloc[0]
    label = "Anomalous" if int(record["anomaly"]) == 1 else "Normal"

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Segment", f"SEG-{int(record['segment']):04d}")
    with kpi_cols[1]:
        st.metric("Dataset label", label)
    with kpi_cols[2]:
        st.metric("Channel", str(record["channel"]))
    with kpi_cols[3]:
        st.metric("Sampling", str(record["sampling"]))

    details_col, chart_col = st.columns([.95, 1.35])
    with details_col:
        section_header("Feature profile", "Observed engineered values")
        feature_table = pd.DataFrame(
            {"Feature": data["feature_names"], "Value": [record[name] for name in data["feature_names"]]}
        )
        st.dataframe(
            feature_table.style.format({"Value": "{:.6g}"}),
            hide_index=True,
            width="stretch",
            height=440,
        )
    with chart_col:
        section_header("Feature context", "Compare with labeled population")
        feature_name = st.selectbox(
            "Feature to visualize",
            data["feature_names"],
            index=data["feature_names"].index("n_peaks"),
        )
        plot_data = records[["anomaly", feature_name]].copy()
        plot_data["Class"] = plot_data["anomaly"].map({0: "Normal", 1: "Anomalous"})
        chart = px.histogram(
            plot_data,
            x=feature_name,
            color="Class",
            nbins=36,
            barmode="overlay",
            opacity=.72,
            color_discrete_map=COLORS,
            title=f"{feature_name} distribution by dataset label",
        )
        chart.add_vline(
            x=float(record[feature_name]),
            line_dash="dash",
            line_color="#f8d278",
            annotation_text="Selected segment",
            annotation_position="top",
        )
        chart.update_layout(**PLOTLY_LAYOUT, height=430)
        st.plotly_chart(chart, width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            "Distributions show observed labeled records, not model predictions or a causal relationship."
        )

    section_header("Raw signal", "Timestamped telemetry trace")
    raw = data["raw_telemetry"].loc[
        data["raw_telemetry"]["segment"].eq(selected_segment)
    ].copy()
    if raw.empty:
        st.info("No raw telemetry samples were found for this segment.")
    else:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
        raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
        trace = px.line(
            raw,
            x="timestamp",
            y="value",
            title=f"Segment {selected_segment} · {len(raw):,} telemetry samples",
            color_discrete_sequence=[COLORS[label]],
        )
        trace.update_layout(**PLOTLY_LAYOUT, height=350)
        trace.update_yaxes(title="Telemetry value")
        trace.update_xaxes(title="Timestamp (UTC)")
        st.plotly_chart(trace, width="stretch", config=PLOTLY_CONFIG)


def render_prediction(data: dict, root_string: str, pipeline) -> None:
    records = data["records"]
    hero(
        "Anomaly Prediction",
        "Score an existing OPS-SAT segment with the saved baseline XGBoost pipeline and review the model’s explanation.",
        kicker="INFERENCE / HUMAN REVIEW",
    )
    callout(
        "This is a machine-learning output based on dataset features. It is not a diagnosis "
        "of a satellite hardware fault and must be reviewed by a domain expert.",
        warning=True,
    )
    segment_ids = records["segment"].astype(int).tolist()
    selected_segment = st.selectbox(
        "Choose an existing telemetry segment",
        segment_ids,
        format_func=lambda value: f"SEG-{value:04d}",
        key="prediction_segment",
    )
    record = records.loc[records["segment"].eq(selected_segment)].iloc[0]
    safe_channel = html.escape(str(record["channel"]))
    prediction = get_segment_prediction(root_string, int(selected_segment))
    actual_label = int(record["anomaly"])
    predicted_class = prediction["predicted_class"]
    is_anomaly = predicted_class == 1

    columns = st.columns([1.1, 1, 1])
    with columns[0]:
        status = "ANOMALOUS" if is_anomaly else "NORMAL"
        status_color = "#ff7185" if is_anomaly else "#55e0bc"
        st.markdown(
            f"""
            <div class="og-card" style="border-color:{status_color}55;">
              <div class="og-card-label">Baseline XGBoost output</div>
              <div class="og-card-value" style="color:{status_color};font-size:2rem;">{status}</div>
              <div class="og-card-note">Segment SEG-{selected_segment:04d} · Channel {safe_channel}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.metric("Anomaly probability", f"{prediction['anomaly_probability']:.1%}")
        st.progress(prediction["anomaly_probability"])
    with columns[2]:
        st.metric("Reference dataset label", "Anomalous (1)" if actual_label else "Normal (0)")
        st.caption("Ground-truth label shown for retrospective dataset inspection.")

    st.markdown(
        f"""
        <div class="og-card">
          <div class="og-card-label">Model score</div>
          <div class="og-mono">P(Normal) {prediction['normal_probability']:.4f}
          &nbsp; / &nbsp; P(Anomalous) {prediction['anomaly_probability']:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_header("Model rationale", "Explain the selected prediction")
    st.write(
        "Global SHAP ranks features across an explained test sample. For a segment-specific "
        "reason, request a local SHAP explanation computed from this saved model and the "
        "same 80-row training background used in Stage 6."
    )
    if st.button("Calculate local SHAP contributions", type="primary"):
        with st.spinner("Explaining this segment with the saved XGBoost model…"):
            try:
                details = explain_selected_segment(
                    pipeline,
                    record,
                    data["feature_names"],
                    data["root"] / "data" / "processed" / "train_processed.csv",
                )
                st.session_state["local_shap"] = {
                    "segment": int(selected_segment),
                    "details": details,
                }
            except Exception as error:
                st.error(f"Could not calculate this explanation: {error}")
    local_result = st.session_state.get("local_shap")
    if local_result and local_result["segment"] == int(selected_segment):
        contributions = local_result["details"]["contributions"].head(8).copy()
        contributions["Direction"] = np.where(
            contributions["shap_value"] >= 0,
            "Raises anomaly score",
            "Lowers anomaly score",
        )
        contribution_chart = px.bar(
            contributions.sort_values("shap_value"),
            x="shap_value",
            y="feature",
            color="Direction",
            orientation="h",
            color_discrete_map={
                "Raises anomaly score": "#ff7185",
                "Lowers anomaly score": "#41d9ec",
            },
            hover_data={"feature_value": ":.5g", "shap_value": ":.5f"},
            title="Local SHAP contributions · class 1 probability",
        )
        contribution_chart.update_layout(**PLOTLY_LAYOUT, height=390)
        contribution_chart.update_xaxes(title="SHAP contribution (probability scale)")
        contribution_chart.update_yaxes(title="")
        st.plotly_chart(
            contribution_chart,
            width="stretch",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            f"SHAP baseline {local_result['details']['base_probability']:.4f}; "
            f"reconstructed model anomaly probability "
            f"{local_result['details']['reconstructed_probability']:.4f}. "
            "Contributions describe model behavior, not physical causation."
        )
    else:
        st.info(
            "Choose “Calculate local SHAP contributions” to generate a real explanation "
            "for this segment. No SHAP values are fabricated or reused from another segment."
        )


def render_performance(data: dict) -> None:
    hero(
        "Model Performance",
        "A transparent comparison of saved model results on the official held-out test split.",
        kicker="VALIDATION / TEST SET",
    )
    evaluation = data["evaluation"]["evaluation"]
    st.caption(
        f"Split: {evaluation['split']} · {evaluation['records']:,} records · "
        f"0 = Normal, 1 = Anomalous · decision threshold = {evaluation['threshold']:.2f}"
    )
    comparison = data["performance"].set_index("model").loc[list(MODEL_ORDER)].reset_index()
    selected_model = st.selectbox(
        "Inspect model metrics",
        MODEL_ORDER,
        index=MODEL_ORDER.index("XGBoost baseline"),
    )
    row = comparison.set_index("model").loc[selected_model]
    columns = st.columns(6)
    metric_fields = (
        ("Accuracy", "accuracy"),
        ("Precision", "precision_anomaly"),
        ("Anomaly recall", "recall_anomaly"),
        ("F1-score", "f1_anomaly"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    )
    for column, (label, key) in zip(columns, metric_fields):
        with column:
            st.metric(label, f"{float(row[key]):.3f}")

    section_header(
        "Model comparison",
        "All saved Stage 4 models",
        "Accuracy is not sufficient with a minority anomaly class; compare anomaly recall and false-negative counts as well.",
    )
    display_columns = [
        "model",
        "accuracy",
        "precision_anomaly",
        "recall_anomaly",
        "f1_anomaly",
        "roc_auc",
        "pr_auc",
        "false_positive",
        "false_negative",
    ]
    st.dataframe(
        comparison[display_columns].style.format(
            {
                "accuracy": "{:.3f}",
                "precision_anomaly": "{:.3f}",
                "recall_anomaly": "{:.3f}",
                "f1_anomaly": "{:.3f}",
                "roc_auc": "{:.3f}",
                "pr_auc": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    predictions = data["predictions"]
    predictions = predictions.loc[predictions["model"].eq(selected_model)]
    matrix = data["evaluation"]["models"][selected_model]["confusion_matrix"]
    left, right = st.columns(2)
    with left:
        heatmap = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted Normal", "Predicted Anomalous"],
                y=["Actual Normal", "Actual Anomalous"],
                text=matrix,
                texttemplate="%{text}",
                colorscale=[[0, "#10263d"], [1, "#31c7e8"]],
                showscale=False,
                hovertemplate="Actual %{y}<br>%{x}: %{z}<extra></extra>",
            )
        )
        heatmap.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{selected_model} · confusion matrix",
            height=390,
        )
        heatmap.update_yaxes(autorange="reversed")
        st.plotly_chart(heatmap, width="stretch", config=PLOTLY_CONFIG)
    with right:
        outcomes = (
            predictions["outcome"]
            .value_counts()
            .reindex(
                ["true_negative", "false_positive", "false_negative", "true_positive"],
                fill_value=0,
            )
            .rename_axis("outcome")
            .reset_index(name="segments")
        )
        outcomes["Outcome"] = outcomes["outcome"].map(
            {
                "true_negative": "True negative",
                "false_positive": "False positive",
                "false_negative": "False negative",
                "true_positive": "True positive",
            }
        )
        outcome_chart = px.bar(
            outcomes,
            x="Outcome",
            y="segments",
            color="Outcome",
            color_discrete_map={
                "True negative": "#41d9ec",
                "False positive": "#ffca76",
                "False negative": "#ff7185",
                "True positive": "#55e0bc",
            },
            title="Prediction outcomes",
        )
        outcome_chart.update_layout(**PLOTLY_LAYOUT, height=390, showlegend=False)
        st.plotly_chart(outcome_chart, width="stretch", config=PLOTLY_CONFIG)

    section_header("Threshold-independent view", "ROC & precision–recall curves")
    curve_left, curve_right = st.columns(2)
    with curve_left:
        roc = go.Figure()
        for model_name in MODEL_ORDER:
            subset = data["predictions"].loc[
                data["predictions"]["model"].eq(model_name)
            ]
            fpr, tpr, _ = roc_curve(subset["actual"], subset["anomaly_probability"])
            auc_value = float(
                data["evaluation"]["models"][model_name]["roc_auc"]
            )
            roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{model_name} · {auc_value:.3f}",
                    line={"color": COLORS[model_name], "width": 2.4},
                )
            )
        roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance",
                line={"color": "#8295aa", "dash": "dash"},
            )
        )
        roc.update_layout(
            **PLOTLY_LAYOUT,
            title="Receiver operating characteristic",
            xaxis_title="False positive rate",
            yaxis_title="Anomaly recall (true positive rate)",
            height=420,
        )
        st.plotly_chart(roc, width="stretch", config=PLOTLY_CONFIG)
    with curve_right:
        pr = go.Figure()
        for model_name in MODEL_ORDER:
            subset = data["predictions"].loc[
                data["predictions"]["model"].eq(model_name)
            ]
            precision, recall, _ = precision_recall_curve(
                subset["actual"], subset["anomaly_probability"]
            )
            auc_value = float(data["evaluation"]["models"][model_name]["pr_auc"])
            pr.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"{model_name} · {auc_value:.3f}",
                    line={"color": COLORS[model_name], "width": 2.4},
                )
            )
        prevalence = float(
            data["evaluation"]["evaluation"]["class_counts"]["1"]
        ) / float(data["evaluation"]["evaluation"]["records"])
        pr.add_hline(
            y=prevalence,
            line_dash="dash",
            line_color="#8295aa",
            annotation_text=f"Test prevalence {prevalence:.1%}",
        )
        pr.update_layout(
            **PLOTLY_LAYOUT,
            title="Precision–recall",
            xaxis_title="Anomaly recall",
            yaxis_title="Precision",
            height=420,
        )
        st.plotly_chart(pr, width="stretch", config=PLOTLY_CONFIG)

    baseline = data["performance"].set_index("model").loc["XGBoost baseline"]
    tuned = data["performance"].set_index("model").loc["Tuned XGBoost"]
    callout(
        "Baseline vs tuned XGBoost: "
        f"recall {baseline['recall_anomaly']:.3f} vs {tuned['recall_anomaly']:.3f}; "
        f"F1 {baseline['f1_anomaly']:.3f} vs {tuned['f1_anomaly']:.3f}; "
        f"PR-AUC {baseline['pr_auc']:.3f} vs {tuned['pr_auc']:.3f}. "
        "The tuned model is not better on every metric."
    )

    false_rows = predictions.loc[
        predictions["outcome"].isin(["false_positive", "false_negative"])
    ].copy()
    false_rows["Error type"] = false_rows["outcome"].map(
        {"false_positive": "False alarm", "false_negative": "Missed anomaly"}
    )
    st.dataframe(
        false_rows[["segment", "actual", "predicted", "anomaly_probability", "Error type"]]
        .sort_values(["Error type", "segment"]),
        hide_index=True,
        width="stretch",
        height=250,
    )
    st.caption("False positives and false negatives are reported by actual segment ID.")


def render_shap(data: dict) -> None:
    hero(
        "SHAP Explainability",
        "See which engineered telemetry features influenced baseline XGBoost outputs in the Stage 6 explanation sample.",
        kicker="XAI / MODEL BEHAVIOR",
    )
    summary = data["shap_summary"]
    st.caption(
        f"Saved Stage 6 method: {summary['shap_method']} · "
        f"{summary['test_rows_explained']} test rows explained · "
        f"{summary['training_background_rows']} training background rows"
    )
    importance = data["shap_importance"].sort_values(
        "mean_absolute_shap", ascending=True
    )
    plot = px.bar(
        importance,
        x="mean_absolute_shap",
        y="feature",
        orientation="h",
        color="mean_signed_shap",
        color_continuous_scale=["#428bff", "#152b43", "#ff7185"],
        hover_data={"mean_signed_shap": ":.5f", "mean_absolute_shap": ":.5f"},
        title="Global feature contribution magnitude",
    )
    plot.update_layout(
        **PLOTLY_LAYOUT,
        height=600,
        xaxis_title="Mean absolute SHAP contribution",
        yaxis_title="",
    )
    st.plotly_chart(plot, width="stretch", config=PLOTLY_CONFIG)
    st.caption(
        "Mean absolute SHAP ranks typical contribution magnitude. Signed SHAP indicates "
        "average direction over the explained sample. Positive raises P(anomaly=1); "
        "negative lowers it."
    )

    summary_figure = data["figure_dir"] / "shap_summary.png"
    dependence_figure = data["figure_dir"] / "shap_dependence.png"
    waterfall_normal = data["figure_dir"] / "shap_normal_prediction.png"
    waterfall_anomalous = data["figure_dir"] / "shap_anomalous_prediction.png"
    left, right = st.columns(2)
    with left:
        section_header("Global", "SHAP beeswarm")
        show_figure(summary_figure, "Saved Stage 6 SHAP summary for class-1 probability.")
    with right:
        section_header("Feature response", "Dependence")
        show_figure(dependence_figure, "Saved Stage 6 dependence plot; model association, not causation.")

    section_header("Local examples", "Saved normal and anomalous explanations")
    normal, anomalous = st.columns(2)
    with normal:
        local = summary["selected_predictions"]["normal"]
        st.markdown(
            f"**Segment {local.get('segment_id', 'unknown')} · Normal**  \n"
            f"P(anomaly) = {local['predicted_anomaly_probability']:.3f}"
        )
        show_figure(waterfall_normal)
    with anomalous:
        local = summary["selected_predictions"]["anomalous"]
        st.markdown(
            f"**Segment {local.get('segment_id', 'unknown')} · Anomalous**  \n"
            f"P(anomaly) = {local['predicted_anomaly_probability']:.3f}"
        )
        show_figure(waterfall_anomalous)

    top = data["shap_importance"].sort_values(
        "mean_absolute_shap", ascending=False
    ).head(5)
    st.dataframe(top, hide_index=True, width="stretch")
    callout(
        summary["interpretation"]["limitation"],
        warning=True,
    )


def render_about(data: dict) -> None:
    records = data["records"]
    raw_sample_count = len(data["raw_telemetry"])
    training_count = int(records["train"].eq(1).sum())
    testing_count = int(records["train"].eq(0).sum())
    hero(
        "About ORBITGUARD",
        "Explainable Satellite Telemetry Anomaly Intelligence — an applied ML portfolio project built around the OPSSAT-AD benchmark.",
        kicker="MISSION / SCOPE & LIMITATIONS",
    )
    left, right = st.columns(2)
    with left:
        section_header("Objective", "From telemetry patterns to reviewable alerts")
        st.markdown(
            """
            ORBITGUARD explores whether engineered OPS-SAT telemetry segment features
            can distinguish dataset-labelled normal and anomalous segments, and makes
            model outputs inspectable through evaluation and SHAP.
            """
        )
        section_header("Dataset", "ESA OPS-SAT · OPSSAT-AD")
        st.markdown(
            f"""
            - **{len(records):,}** labeled segments in `dataset.csv`.
            - **{raw_sample_count:,}** timestamped raw telemetry samples in `segments.csv`.
            - Binary target: `anomaly` (0 = normal, 1 = anomalous).
            - Official benchmark split: {training_count:,} training and {testing_count:,} test segments.
            - Source: [Zenodo record 12588359](https://zenodo.org/records/12588359).
            """
        )
    with right:
        section_header("Technology", "Transparent, lightweight stack")
        st.markdown(
            """
            **Python · Pandas · scikit-learn · XGBoost · SHAP · Streamlit · Plotly**

            The project uses an official split, tree-based model comparisons, a
            saved preprocessing pipeline, held-out evaluation reports, and
            model-level feature explanations.
            """
        )
        section_header("Workflow", "Collection → explainable intelligence")
        st.markdown(
            """
            1. Verify and download the dataset.
            2. Inspect the raw and engineered data.
            3. Preserve the benchmark split and prepare features.
            4. Train tree baselines and XGBoost using training data.
            5. Evaluate on the held-out test split.
            6. Explain baseline XGBoost behavior with SHAP.
            7. Explore saved results and real segment predictions here.
            """
        )
    section_header("Research limitations", "What this dashboard does not claim")
    callout(
        "The system identifies patterns associated with the dataset's labels. It does not "
        "diagnose actual satellite hardware failures, establish physical causation, or "
        "replace flight-operations review. Evaluation is on one supplied benchmark split; "
        "results may not generalize to other missions, channels, or operating conditions.",
        warning=True,
    )


def main() -> None:
    apply_styles()
    page = top_navigation(PAGE_OPTIONS)
    selected_model = "XGBoost baseline"

    root_string = str(PROJECT_ROOT)
    try:
        data = get_dashboard_tables(root_string)
        pipeline = get_model(root_string)
    except Exception as error:
        st.error(f"ORBITGUARD could not load required project artifacts: {error}")
        st.info(
            "Run Stages 1–6 first, and launch the app from this project using "
            "`python -m streamlit run dashboard/app.py`."
        )
        st.stop()

    if page == "Overview":
        render_overview(data, selected_model)
    elif page == "Telemetry":
        render_explorer(data)
    elif page == "Predictions":
        render_prediction(data, root_string, pipeline)
    elif page == "Model Insights":
        performance_tab, shap_tab = st.tabs(
            ["Model performance", "SHAP explainability"]
        )
        with performance_tab:
            render_performance(data)
        with shap_tab:
            render_shap(data)
    else:
        render_about(data)

    st.markdown(
        '<div class="og-footer">ORBITGUARD · DATASET-LABELLED ANOMALY PATTERNS · '
        "EXPLANATIONS DESCRIBE MODEL BEHAVIOR, NOT PHYSICAL CAUSATION</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

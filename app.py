from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_utils import load_dataset
from utils.project_paths import OUTPUTS_DIR


@st.cache_data
def load_reference_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_data
def load_metrics_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metrics_path in OUTPUTS_DIR.glob("*/*/metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "family": metrics_path.parent.parent.name,
                "model_name": metrics_path.parent.name,
                **payload,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["family", "model_name", "rmse", "mae", "r2"])

    return pd.DataFrame(rows).sort_values(by="rmse", ascending=True)


def available_csv_files() -> list[Path]:
    return sorted(OUTPUTS_DIR.rglob("*.csv"))


st.set_page_config(page_title="AqSolDB Course Project Dashboard", layout="wide")

dataset = load_reference_dataset()
metrics_table = load_metrics_table()
csv_files = available_csv_files()

st.title("AqSolDB Course Project Dashboard")
st.caption("Simple dashboard for dataset inspection, model comparisons, tuning runs, and ablation studies.")

summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Dataset Rows", f"{len(dataset):,}")
summary_col2.metric("Tracked Output Tables", len(csv_files))
summary_col3.metric("Completed Model Runs", len(metrics_table))

tab_dataset, tab_results, tab_studies = st.tabs(["Dataset", "Model Results", "Studies"])

with tab_dataset:
    st.subheader("Reference Dataset")
    st.dataframe(dataset[["Name", "SMILES", "Solubility"]], use_container_width=True, height=420)

with tab_results:
    st.subheader("Experiment Leaderboard")
    if metrics_table.empty:
        st.info("Run any script from the `train/` folder to populate model results.")
    else:
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    prediction_files = [path for path in csv_files if path.name == "predictions.csv"]
    if prediction_files:
        selected_predictions = st.selectbox(
            "Prediction Table",
            options=prediction_files,
            format_func=lambda path: f"{path.parent.parent.name}/{path.parent.name}",
        )
        st.dataframe(pd.read_csv(selected_predictions), use_container_width=True, height=320)

with tab_studies:
    st.subheader("Tuning And Ablation Outputs")
    study_files = [path for path in csv_files if path.name != "predictions.csv"]
    if not study_files:
        st.info("Run a script from `tuning/` or `ablation/` to populate study outputs.")
    else:
        selected_study = st.selectbox(
            "Study Table",
            options=study_files,
            format_func=lambda path: str(path.relative_to(OUTPUTS_DIR)),
        )
        st.dataframe(pd.read_csv(selected_study), use_container_width=True, height=360)

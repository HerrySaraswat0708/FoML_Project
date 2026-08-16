import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from utils.data_utils import load_dataset
from utils.project_paths import OUTPUTS_DIR

def load_reference_dataset() -> pd.DataFrame:
    return load_dataset()


def load_metrics_table() -> pd.DataFrame:
    rows = []  # type: List[Dict[str, object]]
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


def available_csv_files() -> List[Path]:
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


def render_dataset_section() -> None:
    st.subheader("Reference Dataset")
    st.dataframe(dataset[["Name", "SMILES", "Solubility"]], height=420)


def render_results_section() -> None:
    st.subheader("Experiment Leaderboard")
    if metrics_table.empty:
        st.info("Run any script from the `train/` folder to populate model results.")
    else:
        st.dataframe(metrics_table)

    prediction_files = [path for path in csv_files if path.name == "predictions.csv"]
    if prediction_files:
        selected_predictions = st.selectbox(
            "Prediction Table",
            options=prediction_files,
            format_func=lambda path: f"{path.parent.parent.name}/{path.parent.name}",
        )
        st.dataframe(pd.read_csv(selected_predictions), height=320)


def render_studies_section() -> None:
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
        st.dataframe(pd.read_csv(selected_study), height=360)


if hasattr(st, "tabs"):
    tab_dataset, tab_results, tab_studies = st.tabs(["Dataset", "Model Results", "Studies"])
    with tab_dataset:
        render_dataset_section()
    with tab_results:
        render_results_section()
    with tab_studies:
        render_studies_section()
else:
    section = st.selectbox("Section", ["Dataset", "Model Results", "Studies"])
    if section == "Dataset":
        render_dataset_section()
    elif section == "Model Results":
        render_results_section()
    else:
        render_studies_section()

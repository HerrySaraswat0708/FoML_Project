import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from rdkit import Chem

from DNN.dense_regressor import DenseRegressor
from GraphML.GraphMP import GraphMP
from utils.data_utils import bond_feature_vector, atom_feature_vector, descriptor_vector, fingerprint_vector, load_dataset
from utils.project_paths import OUTPUTS_DIR

APP_ASSETS_DIR = Path(__file__).resolve().parent / "app_assets"

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


@st.cache_resource
def load_prediction_assets():
    norm = json.loads((APP_ASSETS_DIR / "norm_stats.json").read_text())

    dense_cfg = json.loads((APP_ASSETS_DIR / "dense_config.json").read_text())
    dense_model = DenseRegressor(dense_cfg["input_dim"], tuple(dense_cfg["hidden_layers"]), dense_cfg["dropout"])
    dense_model.load_state_dict(torch.load(APP_ASSETS_DIR / "dense_model.pt", map_location="cpu", weights_only=True))
    dense_model.eval()
    dense_scaler = joblib.load(APP_ASSETS_DIR / "dense_scaler.joblib")

    graph_cfg = json.loads((APP_ASSETS_DIR / "graph_config.json").read_text())
    graph_model = GraphMP(
        in_channels=graph_cfg["in_channels"],
        out_channels=graph_cfg["hidden_channels"],
        global_dim=graph_cfg["global_dim"],
        edge_dim=graph_cfg["edge_dim"],
        dropout=graph_cfg["dropout"],
    )
    graph_model.load_state_dict(torch.load(APP_ASSETS_DIR / "graph_model.pt", map_location="cpu", weights_only=True))
    graph_model.eval()

    return {
        "target_mean": norm["target_mean"],
        "target_std": norm["target_std"],
        "dense_model": dense_model,
        "dense_scaler": dense_scaler,
        "graph_model": graph_model,
    }


def predict_dense(mol, assets: dict) -> float:
    features = np.concatenate([fingerprint_vector(mol), descriptor_vector(mol)]).reshape(1, -1)
    scaled = assets["dense_scaler"].transform(features)
    with torch.no_grad():
        raw = assets["dense_model"](torch.tensor(scaled, dtype=torch.float32)).item()
    return raw * assets["target_std"] + assets["target_mean"]


def predict_graph(mol, assets: dict) -> float:
    from torch_geometric.data import Batch, Data

    x = torch.tensor([atom_feature_vector(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_pairs, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_pairs += [[i, j], [j, i]]
        bf = bond_feature_vector(bond)
        edge_feats += [bf, bf]
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 10), dtype=torch.float)

    global_features = torch.tensor(descriptor_vector(mol), dtype=torch.float).view(1, -1)
    batch = Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr)])
    with torch.no_grad():
        raw = assets["graph_model"](
            batch.x, batch.edge_index, batch.batch, batch.edge_attr, global_features
        ).item()
    return raw * assets["target_std"] + assets["target_mean"]


def solubility_label(logs: float) -> str:
    if logs >= 0:
        return "Very soluble"
    if logs >= -2:
        return "Soluble"
    if logs >= -4:
        return "Moderately soluble"
    if logs >= -6:
        return "Poorly soluble"
    return "Insoluble"


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


def render_predict_section() -> None:
    st.subheader("Predict Solubility From SMILES")
    st.caption("Runs live inference with the saved DenseRegressor and tuned GraphMP checkpoints.")

    example_col, _ = st.columns([3, 1])
    smiles = example_col.text_input(
        "SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O", help="e.g. CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)"
    )
    model_choice = st.radio(
        "Model",
        options=["GraphMP tuned — best model (R²=0.80)", "DenseRegressor (R²=0.74)"],
        horizontal=True,
    )

    if st.button("Predict", type="primary"):
        mol = Chem.MolFromSmiles(smiles.strip()) if smiles else None
        if mol is None:
            st.error(f"Could not parse SMILES: `{smiles}`")
            return
        if mol.GetNumAtoms() == 0:
            st.error("Molecule has no atoms.")
            return

        assets = load_prediction_assets()
        try:
            if model_choice.startswith("GraphMP"):
                if mol.GetNumBonds() == 0:
                    st.warning("GraphMP needs at least one bond — try a larger molecule, or use DenseRegressor.")
                    return
                logs = predict_graph(mol, assets)
            else:
                logs = predict_dense(mol, assets)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.metric("Predicted logS", f"{logs:.3f}", solubility_label(logs))
        st.caption(
            f"Canonical SMILES: `{Chem.MolToSmiles(mol)}` "
            f"· Atoms: {mol.GetNumAtoms()} · Bonds: {mol.GetNumBonds()}"
        )


if hasattr(st, "tabs"):
    tab_dataset, tab_results, tab_studies, tab_predict = st.tabs(["Dataset", "Model Results", "Studies", "Predict"])
    with tab_dataset:
        render_dataset_section()
    with tab_results:
        render_results_section()
    with tab_studies:
        render_studies_section()
    with tab_predict:
        render_predict_section()
else:
    section = st.selectbox("Section", ["Dataset", "Model Results", "Studies", "Predict"])
    if section == "Dataset":
        render_dataset_section()
    elif section == "Model Results":
        render_results_section()
    elif section == "Studies":
        render_studies_section()
    else:
        render_predict_section()

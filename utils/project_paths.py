from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_PATH = DATA_DIR / "curated-solubility-dataset.csv"
FEATURE_MATRIX_PATH = DATA_DIR / "X.npy"
TARGET_VECTOR_PATH = DATA_DIR / "y.npy"
GRAPH_DATASET_PATH = DATA_DIR / "aqsoldb_graph_dataset.pt"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_output_dir(family: str, model_name: str) -> Path:
    return ensure_dir(OUTPUTS_DIR / family / model_name)


def study_output_dir(study_name: str) -> Path:
    return ensure_dir(OUTPUTS_DIR / study_name)


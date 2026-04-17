from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_dense_regressor import train_and_evaluate as train_dense_regressor
from train.train_gaussian_process import train_and_evaluate as train_gaussian_process
from train.train_linear_regression import train_and_evaluate as train_linear_regression
from train.train_mlp_regressor import train_and_evaluate as train_mlp_regressor
from utils.data_utils import load_dataset, save_pca3d_dataset
from utils.project_paths import PCA3D_DATASET_PATH, study_output_dir
from utils.training_utils import save_json, set_global_seed


def build_comparison_rows(summary: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, result in summary.items():
        original = result["original"]
        pca3d = result["pca3d"]
        rows.append(
            {
                "model_name": model_name,
                "original_r2": float(original["r2"]),
                "pca3d_r2": float(pca3d["r2"]),
                "delta_r2": float(pca3d["r2"] - original["r2"]),
                "original_rmse": float(original["rmse"]),
                "pca3d_rmse": float(pca3d["rmse"]),
                "delta_rmse": float(pca3d["rmse"] - original["rmse"]),
                "original_mae": float(original["mae"]),
                "pca3d_mae": float(pca3d["mae"]),
                "delta_mae": float(pca3d["mae"] - original["mae"]),
            }
        )
    return rows


def main(device: str = "auto") -> None:
    output_dir = study_output_dir("pca3d_comparison")
    set_global_seed(42)

    frame = load_dataset()
    save_pca3d_dataset(frame, path=PCA3D_DATASET_PATH, source_feature_mode="combined", n_components=3)

    summary = {
        "linear_family_auto": {
            "original": train_linear_regression(feature_mode="combined", use_best_config=True),
            "pca3d": train_linear_regression(feature_mode="pca3d", use_best_config=True),
        },
        "gaussian_process": {
            "original": train_gaussian_process(feature_mode="descriptor", use_best_config=True),
            "pca3d": train_gaussian_process(feature_mode="pca3d", use_best_config=True),
        },
        "sklearn_mlp_regressor": {
            "original": train_mlp_regressor(feature_mode="combined"),
            "pca3d": train_mlp_regressor(feature_mode="pca3d"),
        },
        "dense_regressor": {
            "original": train_dense_regressor(feature_mode="combined", device=device),
            "pca3d": train_dense_regressor(feature_mode="pca3d", device=device),
        },
    }

    comparison = pd.DataFrame(build_comparison_rows(summary)).sort_values(by="delta_r2", ascending=False)
    comparison.to_csv(output_dir / "pca3d_vs_original_metrics.csv", index=False)
    save_json(output_dir / "pca3d_vs_original_metrics.json", summary)

    print(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Create a 3D PCA dataset and compare model performance against original features.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(device=args.device)

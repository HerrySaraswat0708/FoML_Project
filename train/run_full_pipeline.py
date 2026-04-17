from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_dense_regressor import train_and_evaluate as train_dense_regressor
from train.train_gaussian_process import train_and_evaluate as train_gaussian_process
from train.train_graph_cn import train_and_evaluate as train_graph_cn
from train.train_graph_mp import train_and_evaluate as train_graph_mp
from train.train_graph_net import train_and_evaluate as train_graph_net
from train.train_graph_sage import train_and_evaluate as train_graph_sage
from train.train_linear_regression import train_and_evaluate as train_linear_regression
from train.train_mlp_regressor import train_and_evaluate as train_mlp_regressor


def main(device: str = "auto") -> None:
    summary = {
        "linear_regression": train_linear_regression(),
        "gaussian_process": train_gaussian_process(),
        "mlp_regressor": train_mlp_regressor(),
        "dense_regressor": train_dense_regressor(device=device),
        "graph_cn": train_graph_cn(device=device),
        "graph_net": train_graph_net(device=device),
        "graph_sage": train_graph_sage(device=device),
        "graph_mp": train_graph_mp(device=device),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full training pipeline.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    main(device=args.device)

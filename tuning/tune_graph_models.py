from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GraphML import GraphCN, GraphNET, GraphSAGE
from utils.data_utils import build_graph_dataset, load_dataset
from utils.metrics import regression_metrics
from utils.project_paths import study_output_dir
from utils.training_utils import predict_graph_regressor, set_global_seed, train_graph_regressor


def main() -> None:
    output_dir = study_output_dir("tuning")
    set_global_seed(42)

    frame = load_dataset()
    dataset = build_graph_dataset(frame)
    train_dataset, test_dataset, _, frame_test = train_test_split(dataset, frame, test_size=0.2, random_state=42)
    fit_dataset, val_dataset = train_test_split(train_dataset, test_size=0.15, random_state=42)

    model_builders = {
        "graph_cn": GraphCN,
        "graph_net": GraphNET,
        "graph_sage": GraphSAGE,
    }
    search_space = [
        {"hidden_channels": 32, "learning_rate": 1e-3},
        {"hidden_channels": 64, "learning_rate": 1e-3},
        {"hidden_channels": 128, "learning_rate": 5e-4},
    ]

    rows: list[dict[str, object]] = []
    input_dim = dataset[0].x.shape[1]
    y_test = frame_test["Solubility"].to_numpy(dtype=float)

    for model_name, model_builder in model_builders.items():
        for config in search_space:
            model = model_builder(in_channels=input_dim, out_channels=config["hidden_channels"])
            model, history = train_graph_regressor(
                model=model,
                train_dataset=fit_dataset,
                val_dataset=val_dataset,
                epochs=60,
                batch_size=32,
                learning_rate=float(config["learning_rate"]),
                weight_decay=1e-5,
            )
            y_pred = predict_graph_regressor(model, test_dataset, batch_size=32)
            metrics = regression_metrics(y_test, y_pred)
            rows.append(
                {
                    "model_name": model_name,
                    "hidden_channels": config["hidden_channels"],
                    "learning_rate": config["learning_rate"],
                    "final_val_loss": history[-1]["val_loss"],
                    **metrics,
                }
            )

    results = pd.DataFrame(rows).sort_values(by="rmse")
    results.to_csv(output_dir / "graph_tuning_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GraphML import GraphCN, GraphMP, GraphNET, GraphSAGE
from utils.data_utils import build_graph_dataset, load_dataset
from utils.metrics import regression_metrics
from utils.project_paths import study_output_dir
from utils.training_utils import get_torch_device, predict_graph_regressor, save_json, set_global_seed, train_graph_regressor


def graph_search_space() -> list[dict[str, object]]:
    return [
        {"hidden_channels": 64, "dropout": 0.10, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 32},
        {"hidden_channels": 64, "dropout": 0.15, "learning_rate": 7e-4, "weight_decay": 5e-5, "batch_size": 32},
        {"hidden_channels": 96, "dropout": 0.15, "learning_rate": 5e-4, "weight_decay": 1e-4, "batch_size": 48},
        {"hidden_channels": 128, "dropout": 0.20, "learning_rate": 5e-4, "weight_decay": 1e-4, "batch_size": 64},
    ]


def main(device: str = "auto") -> None:
    output_dir = study_output_dir("tuning")
    set_global_seed(42)
    torch_device = get_torch_device() if device == "auto" else None

    frame = load_dataset()
    dataset = build_graph_dataset(frame)
    train_dataset, test_dataset, _, frame_test = train_test_split(dataset, frame, test_size=0.2, random_state=42)
    fit_dataset, val_dataset = train_test_split(train_dataset, test_size=0.15, random_state=42)

    model_builders = {
        "graph_cn": GraphCN,
        "graph_mp": GraphMP,
        "graph_net": GraphNET,
        "graph_sage": GraphSAGE,
    }
    search_space = graph_search_space()

    rows: list[dict[str, object]] = []
    input_dim = dataset[0].x.shape[1]
    global_dim = int(dataset[0].global_features.shape[0]) if hasattr(dataset[0], "global_features") else 0
    edge_dim = int(dataset[0].edge_attr.shape[1]) if hasattr(dataset[0], "edge_attr") and dataset[0].edge_attr.numel() > 0 else 1
    y_test = frame_test["Solubility"].to_numpy(dtype=float)
    best_row: dict[str, object] | None = None

    for model_name, model_builder in model_builders.items():
        for config in search_space:
            model_kwargs = {
                "in_channels": input_dim,
                "out_channels": config["hidden_channels"],
                "global_dim": global_dim,
                "dropout": float(config["dropout"]),
            }
            if model_name == "graph_mp":
                model_kwargs["edge_dim"] = edge_dim
            model = model_builder(**model_kwargs)
            model, history = train_graph_regressor(
                model=model,
                train_dataset=fit_dataset,
                val_dataset=val_dataset,
                epochs=150,
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
                patience=30,
                device=device,
            )
            y_pred = predict_graph_regressor(model, test_dataset, batch_size=int(config["batch_size"]))
            metrics = regression_metrics(y_test, y_pred)
            best_val_loss = min(epoch["val_loss"] for epoch in history)
            rows.append(
                {
                    "model_name": model_name,
                    "hidden_channels": config["hidden_channels"],
                    "dropout": config["dropout"],
                    "learning_rate": config["learning_rate"],
                    "weight_decay": config["weight_decay"],
                    "batch_size": config["batch_size"],
                    "best_val_loss": best_val_loss,
                    "epochs_ran": len(history),
                    "device": getattr(model, "device_type", torch_device.type if torch_device is not None else device),
                    **metrics,
                }
            )
            if best_row is None or best_val_loss < best_row["best_val_loss"]:
                best_row = rows[-1]

    results = pd.DataFrame(rows).sort_values(by=["best_val_loss", "rmse"])
    results.to_csv(output_dir / "graph_tuning_results.csv", index=False)
    if best_row is not None:
        save_json(output_dir / "best_graph_config.json", best_row)
    print(results.to_string(index=False))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Tune graph models with PyTorch Geometric.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(device=args.device)

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DNN import DenseRegressor, build_mlp_regressor
from utils.data_utils import build_classical_feature_matrix, load_dataset, split_classical_data
from utils.metrics import regression_metrics
from utils.project_paths import study_output_dir
from utils.training_utils import (
    get_torch_device,
    predict_torch_regressor,
    save_json,
    set_global_seed,
    train_torch_regressor,
)


def dense_search_space() -> list[dict[str, object]]:
    return [
        {"hidden_layers": (256, 128), "dropout": 0.10, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 64},
        {"hidden_layers": (256, 128, 64), "dropout": 0.10, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 64},
        {"hidden_layers": (384, 192, 96), "dropout": 0.15, "learning_rate": 7e-4, "weight_decay": 1e-5, "batch_size": 64},
        {"hidden_layers": (512, 256, 128), "dropout": 0.15, "learning_rate": 5e-4, "weight_decay": 5e-5, "batch_size": 128},
        {"hidden_layers": (256, 256, 128), "dropout": 0.20, "learning_rate": 5e-4, "weight_decay": 1e-4, "batch_size": 128},
    ]


def main(device: str = "auto") -> None:
    output_dir = study_output_dir("tuning")
    set_global_seed(42)
    torch_device = get_torch_device() if device == "auto" else None

    frame = load_dataset()
    X, y, clean_frame, _ = build_classical_feature_matrix(frame)
    X_train, X_test, y_train, y_test, _, _ = split_classical_data(X, y, clean_frame)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)
    X_val = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    configs = dense_search_space()

    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None

    for config in configs:
        model = DenseRegressor(
            input_dim=X_fit.shape[1],
            hidden_dims=config["hidden_layers"],
            dropout=float(config["dropout"]),
        )
        model, history = train_torch_regressor(
            model=model,
            train_features=X_fit,
            train_targets=y_fit,
            val_features=X_val,
            val_targets=y_val,
            epochs=120,
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            patience=20,
            device=device,
        )
        y_pred = predict_torch_regressor(model, X_test_scaled)
        metrics = regression_metrics(y_test, y_pred)
        best_val_loss = min(epoch["val_loss"] for epoch in history)
        row = {
            "hidden_layers": json.dumps(list(config["hidden_layers"])),
            "dropout": float(config["dropout"]),
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "batch_size": int(config["batch_size"]),
            "best_val_loss": float(best_val_loss),
            "epochs_ran": len(history),
            "device": getattr(model, "device_type", torch_device.type if torch_device is not None else device),
            **metrics,
        }
        rows.append(row)
        if best_row is None or row["best_val_loss"] < best_row["best_val_loss"]:
            best_row = row
            joblib.dump(scaler, output_dir / "best_dense_scaler.joblib")
            save_json(output_dir / "best_dense_config.json", row)

    sklearn_mlp_rows: list[dict[str, object]] = []
    sklearn_mlp_configs = [
        {"hidden_layer_sizes": (256, 128), "alpha": 1e-5, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (256, 128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (384, 192, 96), "alpha": 1e-4, "learning_rate_init": 5e-4},
        {"hidden_layer_sizes": (512, 256, 128), "alpha": 1e-3, "learning_rate_init": 5e-4},
    ]
    best_sklearn_mlp_row: dict[str, object] | None = None

    for config in sklearn_mlp_configs:
        model = build_mlp_regressor(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            alpha=float(config["alpha"]),
            learning_rate_init=float(config["learning_rate_init"]),
        )
        model.fit(X_fit, y_fit)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test_scaled)
        val_metrics = regression_metrics(y_val, y_val_pred)
        test_metrics = regression_metrics(y_test, y_test_pred)
        row = {
            "model_name": "sklearn_mlp_regressor",
            "hidden_layers": json.dumps(list(config["hidden_layer_sizes"])),
            "dropout": None,
            "learning_rate": float(config["learning_rate_init"]),
            "weight_decay": float(config["alpha"]),
            "batch_size": 32,
            "best_val_loss": float(val_metrics["rmse"]),
            "epochs_ran": None,
            "device": "cpu",
            **test_metrics,
        }
        sklearn_mlp_rows.append(row)
        if best_sklearn_mlp_row is None or row["best_val_loss"] < best_sklearn_mlp_row["best_val_loss"]:
            best_sklearn_mlp_row = row
            save_json(output_dir / "best_sklearn_mlp_config.json", row)

    dense_rows = [{"model_name": "dense_regressor", **row} for row in rows]
    results = pd.DataFrame(dense_rows + sklearn_mlp_rows).sort_values(by=["best_val_loss", "rmse"])
    results.to_csv(output_dir / "dnn_tuning_results.csv", index=False)
    print(results.to_string(index=False))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Tune the PyTorch dense regressor.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(device=args.device)

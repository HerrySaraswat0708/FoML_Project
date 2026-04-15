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

from DNN import DenseRegressor
from utils.data_utils import build_classical_feature_matrix, load_dataset, split_classical_data
from utils.metrics import regression_metrics
from utils.project_paths import study_output_dir
from utils.training_utils import predict_torch_regressor, save_json, set_global_seed, train_torch_regressor


def main() -> None:
    output_dir = study_output_dir("tuning")
    set_global_seed(42)

    frame = load_dataset()
    X, y, clean_frame, _ = build_classical_feature_matrix(frame)
    X_train, X_test, y_train, y_test, _, _ = split_classical_data(X, y, clean_frame)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)
    X_val = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    configs = [
        {"hidden_layers": (128, 64), "dropout": 0.10, "learning_rate": 1e-3},
        {"hidden_layers": (256, 128, 64), "dropout": 0.15, "learning_rate": 1e-3},
        {"hidden_layers": (512, 256, 128), "dropout": 0.20, "learning_rate": 5e-4},
    ]

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
            epochs=80,
            batch_size=64,
            learning_rate=float(config["learning_rate"]),
            weight_decay=1e-5,
        )
        y_pred = predict_torch_regressor(model, X_test_scaled)
        metrics = regression_metrics(y_test, y_pred)
        row = {
            "hidden_layers": json.dumps(list(config["hidden_layers"])),
            "dropout": float(config["dropout"]),
            "learning_rate": float(config["learning_rate"]),
            "final_val_loss": history[-1]["val_loss"],
            **metrics,
        }
        rows.append(row)
        if best_row is None or row["rmse"] < best_row["rmse"]:
            best_row = row
            joblib.dump(scaler, output_dir / "best_dense_scaler.joblib")
            save_json(output_dir / "best_dense_config.json", row)

    results = pd.DataFrame(rows).sort_values(by="rmse")
    results.to_csv(output_dir / "dnn_tuning_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

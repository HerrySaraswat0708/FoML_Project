from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DNN import DenseRegressor
from utils.data_utils import build_classical_feature_matrix, load_dataset, split_classical_data
from utils.training_utils import (
    predict_torch_regressor,
    save_json,
    save_torch_run,
    set_global_seed,
    train_torch_regressor,
)


def train_and_evaluate(
    test_size: float = 0.2,
    random_state: int = 42,
    hidden_layers: tuple[int, ...] = (256, 128, 64),
    dropout: float = 0.15,
    epochs: int = 150,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    feature_mode: str = "combined",
) -> dict[str, float]:
    set_global_seed(random_state)
    frame = load_dataset()
    X, y, clean_frame, feature_names = build_classical_feature_matrix(frame, feature_mode=feature_mode)
    X_train, X_test, y_train, y_test, frame_train, frame_test = split_classical_data(
        X,
        y,
        clean_frame,
        test_size=test_size,
        random_state=random_state,
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=random_state,
        shuffle=True,
    )

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)
    X_val = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = DenseRegressor(
        input_dim=X_fit.shape[1],
        hidden_dims=hidden_layers,
        dropout=dropout,
    )
    model, history = train_torch_regressor(
        model=model,
        train_features=X_fit,
        train_targets=y_fit,
        val_features=X_val,
        val_targets=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    y_pred = predict_torch_regressor(model, X_test_scaled)

    output_dir, metrics = save_torch_run(
        family="dnn",
        model_name="dense_regressor",
        state_dict=model.state_dict(),
        history=history,
        test_frame=frame_test,
        y_test=y_test,
        y_pred=y_pred,
        extra_metadata={
            "feature_mode": feature_mode,
            "num_features": len(feature_names),
            "hidden_layers": list(hidden_layers),
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_rows": len(frame_train),
            "test_rows": len(frame_test),
        },
    )
    joblib.dump(scaler, output_dir / "scaler.joblib")
    save_json(
        output_dir / "model_config.json",
        {
            "input_dim": X_fit.shape[1],
            "hidden_layers": list(hidden_layers),
            "dropout": dropout,
        },
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the PyTorch dense regressor.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--feature-mode",
        choices=["fingerprint", "descriptor", "combined"],
        default="combined",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        test_size=args.test_size,
        random_state=args.random_state,
        hidden_layers=tuple(args.hidden_layers),
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        feature_mode=args.feature_mode,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GraphML import GraphSAGE
from utils.data_utils import build_graph_dataset, load_dataset
from utils.training_utils import predict_graph_regressor, save_torch_run, set_global_seed, train_graph_regressor


def train_and_evaluate(
    test_size: float = 0.2,
    random_state: int = 42,
    hidden_channels: int = 64,
    epochs: int = 120,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
) -> dict[str, float]:
    set_global_seed(random_state)
    frame = load_dataset()
    dataset = build_graph_dataset(frame)

    train_dataset, test_dataset, _, frame_test = train_test_split(
        dataset,
        frame,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    fit_dataset, val_dataset = train_test_split(
        train_dataset,
        test_size=0.15,
        random_state=random_state,
        shuffle=True,
    )

    input_dim = dataset[0].x.shape[1]
    model = GraphSAGE(in_channels=input_dim, out_channels=hidden_channels)
    model, history = train_graph_regressor(
        model=model,
        train_dataset=fit_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    y_pred = predict_graph_regressor(model, test_dataset, batch_size=batch_size)
    y_test = frame_test["Solubility"].to_numpy(dtype=float)

    _, metrics = save_torch_run(
        family="graphml",
        model_name="graph_sage",
        state_dict=model.state_dict(),
        history=history,
        test_frame=frame_test.reset_index(drop=True),
        y_test=y_test,
        y_pred=y_pred,
        extra_metadata={
            "hidden_channels": hidden_channels,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_graphs": len(train_dataset),
            "test_graphs": len(test_dataset),
        },
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the GraphSAGE model.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        test_size=args.test_size,
        random_state=args.random_state,
        hidden_channels=args.hidden_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GraphML import GraphCN, GraphMP, GraphNET, GraphSAGE
from utils.data_utils import build_graph_dataset, load_dataset, make_binary_labels
from utils.metrics import classification_metrics
from utils.project_paths import study_output_dir
from utils.training_utils import (
    predict_graph_binary_classifier,
    set_global_seed,
    train_graph_binary_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run graph binary-classification ablation across all graph architectures.",
    )
    parser.add_argument("--threshold", type=float, default=-3.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = study_output_dir("ablation")
    set_global_seed(args.random_state)
    frame = load_dataset()
    y_binary = make_binary_labels(frame["Solubility"], threshold=args.threshold)
    positive_label = f"Solubility >= {args.threshold}"

    model_builders = {
        "graph_cn": GraphCN,
        "graph_net": GraphNET,
        "graph_sage": GraphSAGE,
        "graph_mp": GraphMP,
    }

    rows: list[dict[str, object]] = []
    for feature_variant in ("atomic_number", "atomic_number_degree", "full"):
        dataset = build_graph_dataset(frame, feature_variant=feature_variant)
        for graph, label in zip(dataset, y_binary):
            graph.y = torch.tensor([float(label)], dtype=torch.float32)

        train_dataset, test_dataset, y_train, y_test, _, _ = train_test_split(
            dataset,
            y_binary,
            frame,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=True,
            stratify=y_binary,
        )
        fit_dataset, val_dataset, _, _ = train_test_split(
            train_dataset,
            y_train,
            test_size=0.15,
            random_state=args.random_state,
            shuffle=True,
            stratify=y_train,
        )

        for model_name, model_builder in model_builders.items():
            model = model_builder(in_channels=dataset[0].x.shape[1], out_channels=64)
            model, _ = train_graph_binary_classifier(
                model=model,
                train_dataset=fit_dataset,
                val_dataset=val_dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            y_pred, y_score = predict_graph_binary_classifier(model, test_dataset, batch_size=args.batch_size)
            metrics = classification_metrics(y_test, y_pred, y_score)
            rows.append(
                {
                    "study_type": "binary_classification",
                    "label_rule": positive_label,
                    "feature_variant": feature_variant,
                    "model_name": model_name,
                    "node_feature_dim": int(dataset[0].x.shape[1]),
                    "positive_rate": float(y_binary.mean()),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows).sort_values(
        by=["f1", "roc_auc", "accuracy"],
        ascending=[False, False, False],
    )
    results.to_csv(output_dir / "graph_classification_ablation.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

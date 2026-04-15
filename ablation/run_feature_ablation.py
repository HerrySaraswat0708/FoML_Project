from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ClassicalModels import (
    build_gaussian_process_classifier,
    build_logistic_regression,
    build_mlp_classifier,
)
from DNN import DenseClassifier
from utils.data_utils import (
    build_classical_feature_matrix,
    load_dataset,
    make_binary_labels,
    split_classical_data,
)
from utils.metrics import classification_metrics
from utils.project_paths import study_output_dir
from utils.training_utils import (
    predict_torch_binary_classifier,
    set_global_seed,
    train_torch_binary_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tabular binary-classification ablation across classical and DNN methods.",
    )
    parser.add_argument("--threshold", type=float, default=-3.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = study_output_dir("ablation")
    set_global_seed(args.random_state)
    frame = load_dataset()

    rows: list[dict[str, object]] = []
    positive_label = f"Solubility >= {args.threshold}"

    for feature_mode in ("fingerprint", "descriptor", "combined"):
        X, y_regression, clean_frame, feature_names = build_classical_feature_matrix(
            frame,
            feature_mode=feature_mode,
        )
        y_binary = make_binary_labels(y_regression, threshold=args.threshold)
        X_train, X_test, y_train, y_test, _, _ = split_classical_data(
            X,
            y_binary,
            clean_frame,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y_binary,
        )

        sklearn_methods = {
            "logistic_regression": build_logistic_regression(),
            "gaussian_process_classifier": build_gaussian_process_classifier(n_restarts_optimizer=0),
            "mlp_classifier": build_mlp_classifier(),
        }

        for model_name, model in sklearn_methods.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]
            metrics = classification_metrics(y_test, y_pred, y_score)
            rows.append(
                {
                    "study_type": "binary_classification",
                    "label_rule": positive_label,
                    "feature_mode": feature_mode,
                    "model_name": model_name,
                    "num_features": len(feature_names),
                    "positive_rate": float(y_binary.mean()),
                    **metrics,
                }
            )

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.15,
            random_state=args.random_state,
            shuffle=True,
            stratify=y_train,
        )
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit)
        X_val = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        dense_model = DenseClassifier(input_dim=X_fit.shape[1], hidden_dims=(256, 128, 64), dropout=0.15)
        dense_model, _ = train_torch_binary_classifier(
            model=dense_model,
            train_features=X_fit,
            train_targets=y_fit,
            val_features=X_val,
            val_targets=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        y_pred, y_score = predict_torch_binary_classifier(dense_model, X_test_scaled)
        metrics = classification_metrics(y_test, y_pred, y_score)
        rows.append(
            {
                "study_type": "binary_classification",
                "label_rule": positive_label,
                "feature_mode": feature_mode,
                "model_name": "dense_classifier",
                "num_features": len(feature_names),
                "positive_rate": float(y_binary.mean()),
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=["f1", "roc_auc", "accuracy"],
        ascending=[False, False, False],
    )
    results.to_csv(output_dir / "tabular_classification_ablation.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

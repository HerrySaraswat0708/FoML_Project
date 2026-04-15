from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ClassicalModels import build_mlp_regressor
from utils.data_utils import build_classical_feature_matrix, load_dataset, split_classical_data
from utils.training_utils import save_sklearn_run, set_global_seed


def train_and_evaluate(
    test_size: float = 0.2,
    random_state: int = 42,
    hidden_layers: tuple[int, ...] = (256, 128, 64),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    feature_mode: str = "combined",
) -> dict[str, float]:
    set_global_seed(random_state)
    frame = load_dataset()
    X, y, clean_frame, feature_names = build_classical_feature_matrix(frame, feature_mode=feature_mode)
    X_train, X_test, y_train, y_test, _, frame_test = split_classical_data(
        X,
        y,
        clean_frame,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_mlp_regressor(
        hidden_layer_sizes=hidden_layers,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    _, metrics = save_sklearn_run(
        family="classical",
        model_name="mlp_regressor",
        model=model,
        test_frame=frame_test,
        y_test=y_test,
        y_pred=y_pred,
        extra_metadata={
            "feature_mode": feature_mode,
            "num_features": len(feature_names),
            "hidden_layers": list(hidden_layers),
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the scikit-learn MLP regressor.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
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
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        feature_mode=args.feature_mode,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

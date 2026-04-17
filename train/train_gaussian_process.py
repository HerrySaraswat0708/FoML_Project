from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ClassicalModels import build_gaussian_process
from utils.data_utils import build_classical_feature_matrix, load_dataset, split_classical_data
from utils.project_paths import OUTPUTS_DIR
from utils.training_utils import save_sklearn_run, set_global_seed


BEST_CLASSICAL_CONFIGS_PATH = OUTPUTS_DIR / "tuning" / "best_classical_configs.json"


def load_best_gpr_config() -> dict[str, object]:
    if not BEST_CLASSICAL_CONFIGS_PATH.exists():
        return {}
    payload = json.loads(BEST_CLASSICAL_CONFIGS_PATH.read_text(encoding="utf-8"))
    return payload.get("gaussian_process", {})


def train_and_evaluate(
    test_size: float = 0.2,
    random_state: int = 42,
    kernel_name: str = "rbf",
    length_scale: float = 1.0,
    matern_nu: float = 1.5,
    rq_alpha: float = 1.0,
    alpha: float = 1e-6,
    feature_mode: str = "descriptor",
    use_best_config: bool = True,
) -> dict[str, float]:
    set_global_seed(random_state)
    if use_best_config:
        best = load_best_gpr_config()
        if best:
            kernel_name = str(best.get("kernel_name", kernel_name))
            length_scale = float(best.get("length_scale", length_scale))
            matern_nu = float(best.get("matern_nu", matern_nu))
            rq_alpha = float(best.get("rq_alpha", rq_alpha))
            alpha = float(best.get("alpha_value", alpha))
            feature_mode = str(best.get("feature_mode", feature_mode))
    frame = load_dataset()
    X, y, clean_frame, feature_names = build_classical_feature_matrix(frame, feature_mode=feature_mode)
    X_train, X_test, y_train, y_test, _, frame_test = split_classical_data(
        X,
        y,
        clean_frame,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_gaussian_process(
        kernel_name=kernel_name,
        length_scale=length_scale,
        matern_nu=matern_nu,
        rq_alpha=rq_alpha,
        alpha=alpha,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    _, metrics = save_sklearn_run(
        family="classical",
        model_name="gaussian_process",
        model=model,
        test_frame=frame_test,
        y_test=y_test,
        y_pred=y_pred,
        extra_metadata={
            "feature_mode": feature_mode,
            "num_features": len(feature_names),
            "kernel_name": kernel_name,
            "length_scale": length_scale,
            "matern_nu": matern_nu,
            "rq_alpha": rq_alpha,
            "alpha": alpha,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Gaussian process regression.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kernel-name", choices=["rbf", "matern", "rational_quadratic"], default="rbf")
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--matern-nu", type=float, default=1.5)
    parser.add_argument("--rq-alpha", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument(
        "--feature-mode",
        choices=["fingerprint", "descriptor", "combined"],
        default="descriptor",
    )
    parser.add_argument("--use-best-config", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        test_size=args.test_size,
        random_state=args.random_state,
        kernel_name=args.kernel_name,
        length_scale=args.length_scale,
        matern_nu=args.matern_nu,
        rq_alpha=args.rq_alpha,
        alpha=args.alpha,
        feature_mode=args.feature_mode,
        use_best_config=bool(args.use_best_config),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

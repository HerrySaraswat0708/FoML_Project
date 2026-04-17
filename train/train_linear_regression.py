from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ClassicalModels import (
    build_elastic_net_regression,
    build_lasso_regression,
    build_linear_regression,
    build_ridge_regression,
)
from utils.data_utils import build_classical_feature_matrix, fit_pca_projection, load_dataset, split_classical_data
from utils.project_paths import OUTPUTS_DIR
from utils.training_utils import save_sklearn_run, set_global_seed


BEST_CLASSICAL_CONFIGS_PATH = OUTPUTS_DIR / "tuning" / "best_classical_configs.json"


def load_best_classical_configs() -> dict[str, dict[str, object]]:
    if not BEST_CLASSICAL_CONFIGS_PATH.exists():
        return {}
    return json.loads(BEST_CLASSICAL_CONFIGS_PATH.read_text(encoding="utf-8"))


def resolve_linear_config(
    use_best_config: bool,
    model_variant: str,
    feature_mode: str,
    alpha: float,
    l1_ratio: float,
    fit_intercept: bool,
    positive: bool,
) -> dict[str, object]:
    config = {
        "model_variant": model_variant,
        "feature_mode": feature_mode,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "fit_intercept": fit_intercept,
        "positive": positive,
    }
    if not use_best_config:
        return config

    best_configs = load_best_classical_configs()
    linear_candidates = {
        "linear": "linear_regression",
        "ridge": "ridge_regression",
        "lasso": "lasso_regression",
        "elastic_net": "elastic_net_regression",
    }
    if model_variant == "auto":
        chosen_key = min(
            (name for name in linear_candidates.values() if name in best_configs),
            key=lambda name: float(best_configs[name]["best_cv_rmse"]),
            default=None,
        )
    else:
        chosen_key = linear_candidates.get(model_variant)

    if chosen_key is None or chosen_key not in best_configs:
        if model_variant == "auto":
            config["model_variant"] = "linear"
        return config

    best = best_configs[chosen_key]
    best_params = json.loads(str(best["best_params"]))
    reverse_variants = {value: key for key, value in linear_candidates.items()}
    config["model_variant"] = reverse_variants[chosen_key]
    if config["feature_mode"] != "pca3d":
        config["feature_mode"] = str(best.get("feature_mode", config["feature_mode"]))
    config["alpha"] = float(best_params.get("model__alpha", config["alpha"]))
    config["l1_ratio"] = float(best_params.get("model__l1_ratio", config["l1_ratio"]))
    config["fit_intercept"] = bool(best_params.get("fit_intercept", best_params.get("model__fit_intercept", config["fit_intercept"])))
    config["positive"] = bool(best_params.get("positive", config["positive"]))
    return config


def train_and_evaluate(
    test_size: float = 0.2,
    random_state: int = 42,
    feature_mode: str = "combined",
    model_variant: str = "auto",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    fit_intercept: bool = True,
    positive: bool = False,
    use_best_config: bool = True,
) -> dict[str, float]:
    set_global_seed(random_state)
    resolved = resolve_linear_config(
        use_best_config=use_best_config,
        model_variant=model_variant,
        feature_mode=feature_mode,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        positive=positive,
    )
    feature_mode = str(resolved["feature_mode"])
    model_variant = str(resolved["model_variant"])
    alpha = float(resolved["alpha"])
    l1_ratio = float(resolved["l1_ratio"])
    fit_intercept = bool(resolved["fit_intercept"])
    positive = bool(resolved["positive"])
    frame = load_dataset()
    source_feature_mode = "combined" if feature_mode == "pca3d" else feature_mode
    X, y, clean_frame, feature_names = build_classical_feature_matrix(frame, feature_mode=source_feature_mode)
    X_train, X_test, y_train, y_test, _, frame_test = split_classical_data(
        X,
        y,
        clean_frame,
        test_size=test_size,
        random_state=random_state,
    )
    pca_explained_variance = None
    if feature_mode == "pca3d":
        X_train, X_test, feature_names, _, pca = fit_pca_projection(X_train, X_test, n_components=3)
        pca_explained_variance = float(sum(pca.explained_variance_ratio_))

    if model_variant == "linear":
        model = build_linear_regression(fit_intercept=fit_intercept, positive=positive)
        model_name = "linear_regression"
    elif model_variant == "ridge":
        model = build_ridge_regression(alpha=alpha, fit_intercept=fit_intercept)
        model_name = "ridge_regression"
    elif model_variant == "lasso":
        model = build_lasso_regression(alpha=alpha, fit_intercept=fit_intercept)
        model_name = "lasso_regression"
    elif model_variant == "elastic_net":
        model = build_elastic_net_regression(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept)
        model_name = "elastic_net_regression"
    else:
        raise ValueError("model_variant must be one of: linear, ridge, lasso, elastic_net")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    _, metrics = save_sklearn_run(
        family="classical",
        model_name=f"{model_name}_pca3d" if feature_mode == "pca3d" else model_name,
        model=model,
        test_frame=frame_test,
        y_test=y_test,
        y_pred=y_pred,
        extra_metadata={
            "model_variant": model_variant,
            "feature_mode": feature_mode,
            "source_feature_mode": source_feature_mode,
            "num_features": len(feature_names),
            "pca_explained_variance": pca_explained_variance,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "fit_intercept": fit_intercept,
            "positive": positive,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the linear regression baseline.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--feature-mode",
        choices=["fingerprint", "descriptor", "combined", "pca3d"],
        default="combined",
    )
    parser.add_argument("--model-variant", choices=["auto", "linear", "ridge", "lasso", "elastic_net"], default="auto")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--fit-intercept", type=int, choices=[0, 1], default=1)
    parser.add_argument("--positive", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use-best-config", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        test_size=args.test_size,
        random_state=args.random_state,
        feature_mode=args.feature_mode,
        model_variant=args.model_variant,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        fit_intercept=bool(args.fit_intercept),
        positive=bool(args.positive),
        use_best_config=bool(args.use_best_config),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

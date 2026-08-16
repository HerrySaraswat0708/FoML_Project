import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, RationalQuadratic, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import build_classical_feature_matrix, fit_pca_projection, load_dataset, split_classical_data
from utils.project_paths import OUTPUTS_DIR
from utils.training_utils import save_sklearn_run, set_global_seed


BEST_CLASSICAL_CONFIGS_PATH = OUTPUTS_DIR / "tuning" / "best_classical_configs.json"


def load_best_gpr_config() -> Dict[str, object]:
    if not BEST_CLASSICAL_CONFIGS_PATH.exists():
        return {}
    payload = json.loads(BEST_CLASSICAL_CONFIGS_PATH.read_text(encoding="utf-8"))
    return payload.get("gaussian_process", {})


def build_fast_gaussian_process(
    kernel_name: str,
    length_scale: float,
    matern_nu: float,
    rq_alpha: float,
    alpha: float,
) -> Pipeline:
    normalized = kernel_name.lower()
    if normalized == "rbf":
        base_kernel = RBF(length_scale=length_scale)
    elif normalized == "matern":
        base_kernel = Matern(length_scale=length_scale, nu=matern_nu)
    elif normalized == "rational_quadratic":
        base_kernel = RationalQuadratic(length_scale=length_scale, alpha=rq_alpha)
    else:
        raise ValueError("kernel_name must be one of: rbf, matern, rational_quadratic")

    # Use the noise term inside the kernel and disable expensive optimizer restarts.
    noise_level = max(float(alpha) * 200.0, 0.2)
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * base_kernel + WhiteKernel(
        noise_level=noise_level,
        noise_level_bounds="fixed",
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=max(float(alpha) * 0.01, 1e-5),
                    normalize_y=True,
                    optimizer=None,
                    n_restarts_optimizer=0,
                    random_state=42,
                ),
            ),
        ]
    )


def choose_representative_subset(
    targets: np.ndarray,
    subset_size: int,
    random_state: int,
    num_bins: int = 10,
) -> np.ndarray:
    if subset_size >= len(targets):
        return np.arange(len(targets))

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(targets, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 2:
        rng = np.random.default_rng(random_state)
        return np.sort(rng.choice(len(targets), size=subset_size, replace=False))

    bucket_ids = np.digitize(targets, bin_edges[1:-1], right=True)
    rng = np.random.default_rng(random_state)
    selected = []
    unique_buckets = np.unique(bucket_ids)
    per_bucket = max(1, subset_size // len(unique_buckets))

    for bucket_id in unique_buckets:
        bucket_indices = np.flatnonzero(bucket_ids == bucket_id)
        take = min(len(bucket_indices), per_bucket)
        if take > 0:
            selected.extend(rng.choice(bucket_indices, size=take, replace=False).tolist())

    if len(selected) < subset_size:
        remaining = np.setdiff1d(np.arange(len(targets)), np.asarray(selected, dtype=int), assume_unique=False)
        extra = rng.choice(remaining, size=subset_size - len(selected), replace=False)
        selected.extend(extra.tolist())
    elif len(selected) > subset_size:
        selected = rng.choice(np.asarray(selected, dtype=int), size=subset_size, replace=False).tolist()

    return np.sort(np.asarray(selected, dtype=int))


def train_and_evaluate(
    test_size: float = 0.2,
    random_state: int = 42,
    kernel_name: str = "rational_quadratic",
    length_scale: float = 2.0,
    matern_nu: float = 1.5,
    rq_alpha: float = 0.3,
    alpha: float = 1e-3,
    feature_mode: str = "descriptor",
    use_best_config: bool = True,
) -> Dict[str, float]:
    set_global_seed(random_state)
    gp_train_subset_size = 500
    if use_best_config:
        best = load_best_gpr_config()
        if best:
            kernel_name = str(best.get("kernel_name", kernel_name))
            matern_nu = float(best.get("matern_nu", matern_nu))
            alpha = float(best.get("alpha_value", alpha))
            if feature_mode != "pca3d":
                feature_mode = str(best.get("feature_mode", feature_mode))
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

    subset_indices = choose_representative_subset(
        targets=y_train,
        subset_size=min(gp_train_subset_size, len(X_train)),
        random_state=random_state,
    )
    X_train_subset = X_train[subset_indices]
    y_train_subset = y_train[subset_indices]

    model = build_fast_gaussian_process(
        kernel_name=kernel_name,
        length_scale=length_scale,
        matern_nu=matern_nu,
        rq_alpha=rq_alpha,
        alpha=alpha,
    )
    model.fit(X_train_subset, y_train_subset)
    y_pred = model.predict(X_test)

    _, metrics = save_sklearn_run(
        family="classical",
        model_name="gaussian_process_pca3d" if feature_mode == "pca3d" else "gaussian_process",
        model=model,
        test_frame=frame_test,
        y_test=y_test,
        y_pred=y_pred,
        extra_metadata={
            "feature_mode": feature_mode,
            "source_feature_mode": source_feature_mode,
            "num_features": len(feature_names),
            "pca_explained_variance": pca_explained_variance,
            "kernel_name": kernel_name,
            "length_scale": length_scale,
            "matern_nu": matern_nu,
            "rq_alpha": rq_alpha,
            "alpha": alpha,
            "train_rows": len(X_train),
            "gp_train_rows": len(X_train_subset),
            "test_rows": len(X_test),
            "approximation_strategy": "representative_subset_exact_gp",
        },
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Gaussian process regression.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kernel-name", choices=["rbf", "matern", "rational_quadratic"], default="rational_quadratic")
    parser.add_argument("--length-scale", type=float, default=2.0)
    parser.add_argument("--matern-nu", type=float, default=1.5)
    parser.add_argument("--rq-alpha", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument(
        "--feature-mode",
        choices=["fingerprint", "descriptor", "combined", "pca3d"],
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

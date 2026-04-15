from __future__ import annotations

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def build_gaussian_process(
    length_scale: float = 1.0,
    alpha: float = 1e-6,
    n_restarts_optimizer: int = 2,
) -> GaussianProcessRegressor:
    kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=42,
    )


def build_gaussian_process_classifier(
    length_scale: float = 1.0,
    n_restarts_optimizer: int = 2,
) -> GaussianProcessClassifier:
    kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    return GaussianProcessClassifier(
        kernel=kernel,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=42,
    )

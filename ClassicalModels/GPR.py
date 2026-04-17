from __future__ import annotations

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_gpr_kernel(
    kernel_name: str = "rbf",
    length_scale: float = 1.0,
    matern_nu: float = 1.5,
    rq_alpha: float = 1.0,
):
    normalized = kernel_name.lower()
    if normalized == "rbf":
        return RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    if normalized == "matern":
        return Matern(length_scale=length_scale, nu=matern_nu, length_scale_bounds=(1e-2, 1e2))
    if normalized == "rational_quadratic":
        return RationalQuadratic(length_scale=length_scale, alpha=rq_alpha)
    raise ValueError("kernel_name must be one of: rbf, matern, rational_quadratic")


def build_gaussian_process(
    kernel_name: str = "rbf",
    length_scale: float = 1.0,
    matern_nu: float = 1.5,
    rq_alpha: float = 1.0,
    alpha: float = 1e-6,
    n_restarts_optimizer: int = 2,
) -> Pipeline:
    kernel = build_gpr_kernel(
        kernel_name=kernel_name,
        length_scale=length_scale,
        matern_nu=matern_nu,
        rq_alpha=rq_alpha,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    normalize_y=True,
                    n_restarts_optimizer=n_restarts_optimizer,
                    random_state=42,
                ),
            ),
        ]
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

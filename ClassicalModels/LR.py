from __future__ import annotations

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_linear_regression(
    fit_intercept: bool = True,
    positive: bool = False,
) -> LinearRegression:
    return LinearRegression(fit_intercept=fit_intercept, positive=positive)


def build_ridge_regression(
    alpha: float = 1.0,
    fit_intercept: bool = True,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)),
        ]
    )


def build_lasso_regression(
    alpha: float = 1e-3,
    fit_intercept: bool = True,
    max_iter: int = 5000,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, random_state=42)),
        ]
    )


def build_elastic_net_regression(
    alpha: float = 1e-3,
    l1_ratio: float = 0.5,
    fit_intercept: bool = True,
    max_iter: int = 5000,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=fit_intercept,
                    max_iter=max_iter,
                    random_state=42,
                ),
            ),
        ]
    )


def build_logistic_regression(max_iter: int = 1000) -> LogisticRegression:
    return LogisticRegression(max_iter=max_iter, random_state=42)

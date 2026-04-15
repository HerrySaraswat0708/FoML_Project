from __future__ import annotations

from sklearn.linear_model import LinearRegression, LogisticRegression


def build_linear_regression() -> LinearRegression:
    return LinearRegression()


def build_logistic_regression(max_iter: int = 1000) -> LogisticRegression:
    return LogisticRegression(max_iter=max_iter, random_state=42)

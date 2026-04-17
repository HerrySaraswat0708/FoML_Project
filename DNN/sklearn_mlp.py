from __future__ import annotations

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mlp_regressor(
    hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    max_iter: int = 250,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=32,
                    learning_rate="adaptive",
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                ),
            ),
        ]
    )


def build_mlp_classifier(
    hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    max_iter: int = 250,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=32,
                    learning_rate="adaptive",
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                ),
            ),
        ]
    )

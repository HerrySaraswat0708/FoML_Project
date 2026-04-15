from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import RandomizedSearchCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ClassicalModels import build_gaussian_process, build_linear_regression, build_mlp_regressor
from utils.data_utils import build_classical_feature_matrix, load_dataset, split_classical_data
from utils.metrics import regression_metrics
from utils.project_paths import study_output_dir


def main() -> None:
    output_dir = study_output_dir("tuning")
    frame = load_dataset()
    X, y, clean_frame, _ = build_classical_feature_matrix(frame)
    X_train, X_test, y_train, y_test, _, _ = split_classical_data(X, y, clean_frame)

    rows: list[dict[str, object]] = []

    baseline = build_linear_regression()
    baseline.fit(X_train, y_train)
    baseline_metrics = regression_metrics(y_test, baseline.predict(X_test))
    rows.append({"model_name": "linear_regression", "search_type": "baseline", **baseline_metrics})

    gpr_search = RandomizedSearchCV(
        estimator=build_gaussian_process(),
        param_distributions={
            "alpha": [1e-6, 1e-4, 1e-3, 1e-2],
            "kernel": [RBF(length_scale=value, length_scale_bounds=(1e-2, 1e2)) for value in (0.5, 1.0, 2.0, 5.0)],
        },
        n_iter=6,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=1,
    )
    gpr_search.fit(X_train, y_train)
    gpr_metrics = regression_metrics(y_test, gpr_search.best_estimator_.predict(X_test))
    rows.append(
        {
            "model_name": "gaussian_process",
            "search_type": "randomized_search",
            "best_params": json.dumps(gpr_search.best_params_, default=str),
            **gpr_metrics,
        }
    )

    mlp_search = RandomizedSearchCV(
        estimator=build_mlp_regressor(),
        param_distributions={
            "model__hidden_layer_sizes": [(128, 64), (256, 128, 64), (512, 256, 128)],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [5e-4, 1e-3, 2e-3],
        },
        n_iter=6,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=1,
    )
    mlp_search.fit(X_train, y_train)
    mlp_metrics = regression_metrics(y_test, mlp_search.best_estimator_.predict(X_test))
    rows.append(
        {
            "model_name": "mlp_regressor",
            "search_type": "randomized_search",
            "best_params": json.dumps(mlp_search.best_params_, default=str),
            **mlp_metrics,
        }
    )

    results = pd.DataFrame(rows).sort_values(by="rmse")
    results.to_csv(output_dir / "classical_tuning_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

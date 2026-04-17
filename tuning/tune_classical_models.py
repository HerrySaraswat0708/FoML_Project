from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ClassicalModels import (
    build_elastic_net_regression,
    build_gaussian_process,
    build_lasso_regression,
    build_linear_regression,
    build_ridge_regression,
)
from utils.data_utils import build_classical_feature_matrix, load_dataset, split_classical_data
from utils.metrics import regression_metrics
from utils.project_paths import study_output_dir
from utils.training_utils import save_json, set_global_seed


def evaluate_search(search, X_train, y_train, X_test, y_test) -> dict[str, object]:
    search.fit(X_train, y_train)
    y_pred = search.best_estimator_.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    return {
        "best_params": json.dumps(search.best_params_, default=str),
        "best_cv_rmse": float(-search.best_score_),
        **metrics,
    }


def serialize_gpr_kernel(kernel) -> dict[str, object]:
    if isinstance(kernel, RBF):
        return {
            "kernel_name": "rbf",
            "length_scale": float(kernel.length_scale),
        }
    if isinstance(kernel, Matern):
        return {
            "kernel_name": "matern",
            "length_scale": float(kernel.length_scale),
            "matern_nu": float(kernel.nu),
        }
    if isinstance(kernel, RationalQuadratic):
        return {
            "kernel_name": "rational_quadratic",
            "length_scale": float(kernel.length_scale),
            "rq_alpha": float(kernel.alpha),
        }
    raise ValueError(f"Unsupported GPR kernel type: {type(kernel)!r}")


def main() -> None:
    output_dir = study_output_dir("tuning")
    set_global_seed(42)

    frame = load_dataset()
    feature_sets = {
        mode: build_classical_feature_matrix(frame, feature_mode=mode)[:3]
        for mode in ("descriptor", "combined")
    }

    rows: list[dict[str, object]] = []
    best_configs: dict[str, dict[str, object]] = {}

    linear_searches = {
        "linear_regression": GridSearchCV(
            estimator=build_linear_regression(),
            param_grid={
                "fit_intercept": [True, False],
                "positive": [False, True],
            },
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        ),
        "ridge_regression": GridSearchCV(
            estimator=build_ridge_regression(),
            param_grid={
                "model__alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
                "model__fit_intercept": [True, False],
            },
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        ),
        "lasso_regression": GridSearchCV(
            estimator=build_lasso_regression(),
            param_grid={
                "model__alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                "model__fit_intercept": [True, False],
            },
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        ),
        "elastic_net_regression": GridSearchCV(
            estimator=build_elastic_net_regression(),
            param_grid={
                "model__alpha": [1e-4, 1e-3, 1e-2],
                "model__l1_ratio": [0.2, 0.5, 0.8],
                "model__fit_intercept": [True, False],
            },
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        ),
    }

    for feature_mode, (X, y, clean_frame) in feature_sets.items():
        X_train, X_test, y_train, y_test, _, _ = split_classical_data(X, y, clean_frame)
        for model_name, search in linear_searches.items():
            result = evaluate_search(search, X_train, y_train, X_test, y_test)
            row = {
                "model_name": model_name,
                "feature_mode": feature_mode,
                "search_type": "grid_search",
                **result,
            }
            rows.append(row)
            existing = best_configs.get(model_name)
            if existing is None or row["best_cv_rmse"] < existing["best_cv_rmse"]:
                best_configs[model_name] = row

    X_desc, y_desc, clean_desc = feature_sets["descriptor"]
    X_train, X_test, y_train, y_test, _, _ = split_classical_data(X_desc, y_desc, clean_desc)
    gpr_search = GridSearchCV(
        estimator=build_gaussian_process(n_restarts_optimizer=0),
        param_grid={
            "model__alpha": [1e-6, 1e-4, 1e-3],
            "model__kernel": [
                RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2)),
                RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)),
                RationalQuadratic(length_scale=1.0, alpha=1.0),
            ],
        },
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    gpr_result = evaluate_search(gpr_search, X_train, y_train, X_test, y_test)
    best_gpr_model = gpr_search.best_estimator_.named_steps["model"]
    gpr_row = {
        "model_name": "gaussian_process",
        "feature_mode": "descriptor",
        "search_type": "grid_search",
        **serialize_gpr_kernel(best_gpr_model.kernel),
        "alpha_value": float(best_gpr_model.alpha),
        **gpr_result,
    }
    rows.append(gpr_row)
    best_configs["gaussian_process"] = gpr_row

    results = pd.DataFrame(rows).sort_values(by=["best_cv_rmse", "rmse"])
    results.to_csv(output_dir / "classical_tuning_results.csv", index=False)
    save_json(output_dir / "best_classical_configs.json", best_configs)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

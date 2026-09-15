# Results Snapshot

This folder is a small, git-tracked snapshot of `outputs/` (which is gitignored
because it holds full model weights, multi-hundred-MB Gaussian Process models,
and complete prediction tables). It exists so results are visible directly on
GitHub without anyone having to re-run the pipeline.

- `leaderboard.csv` — every trained model's metrics (RMSE, MAE, R², and the
  hyperparameters it was trained with) in one table, sorted by RMSE.
- `<family>/<model_name>/metrics.json` — the full metrics file for every
  trained model (regenerate with the scripts in `train/`).
- `<family>/<model_name>/predictions_sample.csv` — the first 100 held-out
  predictions (actual vs. predicted solubility, residual, absolute error)
  for the top model in each family: `graph_mp_tuned`, `dense_regressor`,
  `linear_regression`.

Full predictions, saved model weights, and training histories for every run
are in `outputs/` locally (gitignored) and can be regenerated with
`python train/run_full_pipeline.py --device auto`.

## Current Leaderboard

| Rank | Family | Model | RMSE | MAE | R² |
|---:|---|---|---:|---:|---:|
| 1 | graphml | graph_mp_tuned | 1.0471 | 0.7435 | 0.7978 |
| 2 | graphml | graph_sage | 1.0766 | 0.7701 | 0.7862 |
| 3 | graphml | graph_net | 1.0992 | 0.7838 | 0.7771 |
| 4 | graphml | graph_mp | 1.1178 | 0.7870 | 0.7695 |
| 5 | dnn | sklearn_mlp_regressor | 1.1618 | 0.8117 | 0.7512 |
| 6 | dnn | dense_regressor | 1.1771 | 0.8473 | 0.7446 |
| 7 | graphml | graph_cn | 1.1943 | 0.8670 | 0.7369 |
| 8 | classical | mlp_regressor | 1.2088 | 0.8517 | 0.7305 |
| 9 | classical | gaussian_process | 1.2558 | 0.8936 | 0.7091 |
| 10 | classical | lasso_regression | 1.3714 | 1.0245 | 0.6531 |
| 11 | classical | ridge_regression | 1.4308 | 1.0396 | 0.6224 |
| 12 | classical | linear_regression | 1.4616 | 1.0629 | 0.6062 |

(PCA3D-feature variants are excluded here — see `leaderboard.csv` for the full table including them.)

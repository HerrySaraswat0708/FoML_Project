# AqSolDB Course Project

This version of the repository is intentionally simple.

The goal is to show:

- classical regression models on molecular features
- deep neural networks on tabular molecular features
- graph neural networks on molecular graphs
- a clear end-to-end pipeline from data loading to saved results
- hyperparameter tuning and ablation studies in separate folders

The main training pipeline is regression.

The ablation folder is used for a separate binary classification study, where the continuous solubility target is thresholded into two classes.

## Project Structure

```text
AqSolDB/
|-- ClassicalModels/   # builders for regression baselines and their classification-study variants
|-- GraphML/           # graph neural network model definitions
|-- DNN/               # dense neural network model definitions
|-- train/             # end-to-end training scripts for each model
|-- tuning/            # hyperparameter search scripts
|-- ablation/          # ablation-study scripts
|-- utils/             # shared helpers for data, features, metrics, and training loops
|-- outputs/           # metrics, predictions, saved models, tuning logs, ablation results
|-- data/              # dataset and cached feature/graph files
|-- app.py             # simple Streamlit dashboard for dataset and saved results
|-- requirements.txt
```

## Main Workflow

### 1. Regenerate cached inputs if needed

```powershell
python utils\make_data.py
python utils\make_graph_data.py
```

### 2. Train individual models

Classical baselines:

```powershell
python train\train_linear_regression.py
python train\train_gaussian_process.py
```

Deep neural network:

```powershell
python train\train_dense_regressor.py
python train\train_mlp_regressor.py
```

Graph models:

```powershell
python train\train_graph_cn.py
python train\train_graph_net.py
python train\train_graph_sage.py
python train\train_graph_mp.py
```

Run the whole training pipeline:

```powershell
python train\run_full_pipeline.py
```

### 3. Run hyperparameter tuning

```powershell
python tuning\tune_classical_models.py
python tuning\tune_dnn.py
python tuning\tune_graph_models.py
```

### 4. Run ablation studies

These scripts run binary classification experiments, not regression.
By default they label a molecule as positive when `Solubility >= -3.0`.

```powershell
python ablation\run_feature_ablation.py
python ablation\run_graph_ablation.py
```

### 5. Inspect saved outputs

```powershell
streamlit run app.py
```

## Output Layout

Training scripts save outputs in this pattern:

```text
outputs/
|-- classical/
|   |-- linear_regression/
|   |-- ridge_regression/
|   |-- lasso_regression/
|   |-- elastic_net_regression/
|   |-- gaussian_process/
|-- dnn/
|   |-- dense_regressor/
|   |-- sklearn_mlp_regressor/
|-- graphml/
|   |-- graph_cn/
|   |-- graph_net/
|   |-- graph_sage/
|   |-- graph_mp/
|-- tuning/
|-- ablation/
```

Each model folder stores:

- `metrics.json`
- `predictions.csv`
- saved model weights
- training history for neural models

## Notes

- `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, and `GaussianProcessRegressor` are the main classical regression baselines.
- `DenseRegressor` and scikit-learn `MLPRegressor` live under `DNN/` as the tabular neural-network baselines.
- `GraphCN`, `GraphNET`, `GraphSAGE`, and `GraphMP` are graph-level regression models in the main pipeline.
- The ablation scripts convert the problem into binary classification for all families so you can compare tabular and graph methods on the same discrete target.
- The project favors readability over packaging complexity so it is easier to explain in a course report or viva.

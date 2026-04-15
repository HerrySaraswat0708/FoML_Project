# AqSolDB Study Notes

## 1. What This Project Is Doing

This project studies aqueous solubility prediction from SMILES strings. It compares three learning families on the same chemistry problem:

| Family | Input representation | Main task |
|---|---|---|
| Classical ML | Fixed-length tabular vector | Regression |
| Tabular DNN | Fixed-length tabular vector | Regression |
| Graph neural networks | Molecular graph | Regression |

The main pipeline is regression because `Solubility` is continuous. The `ablation/` folder turns the task into binary classification to study how behavior changes when the target is thresholded.

The repository is really about four things:

1. how molecules are represented
2. how different algorithms learn
3. how complexity changes across models
4. what files and results are produced after each run

## 2. Repository Map

| Path | Purpose |
|---|---|
| `data/` | Raw AqSolDB CSV plus cached arrays/graph dataset |
| `utils/` | Cleaning, feature generation, metrics, train loops, output paths |
| `ClassicalModels/` | Builders for linear/logistic regression, Gaussian process, and MLP models |
| `DNN/` | PyTorch dense regressor and classifier |
| `GraphML/` | PyTorch Geometric graph models |
| `train/` | Regression entrypoints |
| `tuning/` | Hyperparameter search scripts |
| `ablation/` | Binary classification studies |
| `outputs/` | Metrics, predictions, saved models, study tables |
| `app.py` | Streamlit dashboard |

## 3. End-To-End Flow

The core pipeline is:

1. load `data/curated-solubility-dataset.csv`
2. clean invalid rows
3. convert molecules into tabular vectors or graphs
4. split into train and test
5. for neural models, split train again into fit and validation
6. train the chosen model
7. evaluate on held-out test data
8. save metrics, predictions, and model artifacts under `outputs/`

Main code paths:

- `utils/data_utils.py`
- `utils/training_utils.py`
- one script from `train/`, `tuning/`, or `ablation/`

Important implementation detail:

- `utils/make_data.py` and `utils/make_graph_data.py` create caches
- the training scripts do not load those caches directly
- they rebuild features and graphs from the cleaned CSV each run

So the cache files are convenience artifacts, not mandatory runtime inputs.

## 4. Data Cleaning And Representation

### 4.1 Cleaned dataset

The raw CSV currently has **9982 rows**.  
The cached tabular arrays show **9980 usable molecules** after cleaning.

`load_dataset()` does the following:

- reads the CSV
- checks that `Name`, `SMILES`, and `Solubility` exist
- coerces `Solubility` to numeric
- drops invalid target rows
- parses each SMILES with RDKit
- keeps only valid molecules
- adds a `CanonicalSMILES` column

From the cached `data/y.npy`:

| Statistic | Value |
|---|---:|
| Clean rows | 9980 |
| Mean solubility | -2.8905 |
| Standard deviation | 2.3679 |
| Minimum | -13.1719 |
| Maximum | 2.1377 |

### 4.2 Tabular representation

For classical models and dense neural nets, each molecule becomes a fixed-length vector:

- Morgan fingerprint, radius `2`, size `1024`
- 12 RDKit descriptors

Default descriptor list:

- `NumHDonors`
- `TPSA`
- `NumRotatableBonds`
- `BertzCT`
- `RingCount`
- `NumAromaticRings`
- `NumValenceElectrons`
- `LabuteASA`
- `HeavyAtomCount`
- `MolWt`
- `MolMR`
- `MolLogP`

Feature modes:

| Mode | Dimension | Meaning |
|---|---:|---|
| `fingerprint` | 1024 | Morgan bits only |
| `descriptor` | 12 | RDKit descriptors only |
| `combined` | 1036 | fingerprint + descriptors |

Cached tabular shapes:

| Object | Shape |
|---|---|
| `X.npy` | `(9980, 1036)` |
| `y.npy` | `(9980,)` |

Math intuition:

- fingerprints capture local substructure patterns
- descriptors capture global physicochemical properties
- the combined vector gives both local and global information

### 4.3 Graph representation

For graph models, each molecule becomes a graph:

- nodes = atoms
- edges = bonds
- graph label = molecule solubility

Default node features:

- atomic number
- atom degree
- aromaticity flag

Graph feature variants used in ablation:

| Variant | Node feature dimension |
|---|---:|
| `atomic_number` | 1 |
| `atomic_number_degree` | 2 |
| `full` | 3 |

Bond weights are encoded as edge attributes:

- single -> `1.0`
- double -> `2.0`
- triple -> `3.0`
- other/aromatic fallback -> `1.5`

Important code fact:

- `build_graph_dataset()` stores `edge_attr`
- the current graph models do **not** use `edge_attr`

So bond weights are prepared, but ignored by the current message-passing layers.

Cached graph dataset statistics from `data/aqsoldb_graph_dataset.pt`:

| Statistic | Value |
|---|---:|
| Graphs | 9980 |
| Node feature dimension | 3 |
| Average atoms per molecule | 17.3830 |
| Median atoms per molecule | 15 |
| Average directed edges per molecule | 35.2669 |
| Median directed edges per molecule | 30 |
| Minimum atoms | 1 |
| Maximum atoms | 388 |
| Graphs with zero bonds | 149 |

## 5. Splits, Losses, Metrics, And Artifacts

Default regression split:

- train rows/graphs = **7984**
- test rows/graphs = **1996**

For dense and graph neural models, the training set is split again into:

- fit = **6786**
- validation = **1198**
- test = **1996**

For binary classification ablation, the splits are stratified by class label.

Regression loss for neural models:

```text
MSE = (1/n) * sum_i (y_i - y_hat_i)^2
```

Saved regression metrics:

```text
RMSE = sqrt((1/n) * sum_i (y_i - y_hat_i)^2)
MAE  = (1/n) * sum_i |y_i - y_hat_i|
R^2  = 1 - sum_i (y_i - y_hat_i)^2 / sum_i (y_i - mean(y))^2
```

Binary classification label rule:

```text
label = 1 if Solubility >= -3.0 else 0
```

Positive class rate from the cached targets:

| Threshold | Positive rate |
|---|---:|
| `Solubility >= -3.0` | 0.5696 |

Neural classifiers use `BCEWithLogitsLoss`, and predictions are thresholded at probability `0.5`.

Saved classification metrics:

- accuracy
- precision
- recall
- F1
- ROC-AUC

Common saved artifacts:

- scikit-learn runs -> `model.joblib`, `metrics.json`, `predictions.csv`
- torch runs -> `model.pt`, `metrics.json`, `predictions.csv`, `history.csv`
- dense regressor also saves `scaler.joblib` and `model_config.json`

`predictions.csv` keeps the original held-out molecule metadata and adds:

- `actual_Solubility`
- `predicted_Solubility`
- `residual`
- `absolute_error`

## 6. Shared Training Logic

Reusable training logic lives in `utils/training_utils.py`.

### 6.1 Reproducibility

`set_global_seed()` seeds:

- Python `random`
- NumPy
- PyTorch CPU and CUDA

### 6.2 Neural training loop behavior

Dense and graph training loops follow the same pattern:

1. move the model to GPU if available
2. train with Adam
3. compute validation loss every epoch
4. keep a copy of the best validation checkpoint
5. restore the best checkpoint at the end

Important behavioral detail:

- the code keeps the **best validation weights**
- it does **not** stop early
- it still completes all epochs, then reloads the best state

### 6.3 Graph-level pooling

All graph models perform graph-level prediction using mean pooling:

```text
h_G = (1 / |V|) * sum_{v in G} h_v
```

This is `global_mean_pool`, which gives one vector per molecule before the final prediction head.

## 7. Algorithms In The Project

### 7.1 Linear Regression

**Implemented in:** `ClassicalModels/LR.py`  
**Trained by:** `train/train_linear_regression.py`

Model equation:

```text
y_hat = w^T x + b
```

Theory:

- prediction is a weighted sum of features
- every feature contributes additively
- nonlinear interactions are not learned directly

Parameter count with default input size:

```text
1036 weights + 1 bias = 1037
```

Complexity:

- training: approximately `O(n d^2)` for least-squares style solving
- inference per sample: `O(d)`
- memory: `O(d)`

Behavior in this project:

- fastest and most interpretable regression baseline
- works reasonably well because the input features already encode chemistry
- cannot model nonlinear structure

Outputs after running:

- `outputs/classical/linear_regression/model.joblib`
- `outputs/classical/linear_regression/metrics.json`
- `outputs/classical/linear_regression/predictions.csv`

### 7.2 Logistic Regression

**Implemented in:** `ClassicalModels/LR.py`  
**Used by:** `ablation/run_feature_ablation.py`

Model equation:

```text
p(y=1 | x) = sigmoid(w^T x + b)
```

This is the binary-classification counterpart of linear regression.

Complexity:

- training: roughly `O(T n d)` for iterative optimization
- inference: `O(d)`

Behavior:

- strong linear classification baseline
- useful for measuring how much nonlinear models help

### 7.3 Gaussian Process Regression

**Implemented in:** `ClassicalModels/GPR.py`  
**Trained by:** `train/train_gaussian_process.py`

Kernel:

```text
k(x, x') = exp(-||x - x'||^2 / (2 l^2))
```

This is the RBF kernel with length scale `l`.

Theory:

- Gaussian processes define a distribution over functions
- similar feature vectors are expected to have similar targets
- the kernel measures similarity between molecules

Implementation details:

- kernel = `RBF`
- `normalize_y=True`
- default `alpha = 1e-6`
- default `n_restarts_optimizer = 2`

Complexity:

```text
training time ~ O(n^3)
memory        ~ O(n^2)
```

With `n = 7984` training samples:

```text
kernel matrix size = 7984 x 7984 = 63,744,256 entries
float64 storage for that matrix alone is about 0.51 GB
```

Behavior in this project:

- mathematically elegant nonlinear baseline
- expensive because of dense kernel matrix factorization
- the current saved model artifact is about **543 MB**

Outputs after running:

- `outputs/classical/gaussian_process/model.joblib`
- `outputs/classical/gaussian_process/metrics.json`
- `outputs/classical/gaussian_process/predictions.csv`

### 7.4 Gaussian Process Classifier

**Implemented in:** `ClassicalModels/GPR.py`  
**Used by:** `ablation/run_feature_ablation.py`

This is the binary-classification version of the GP idea.

Behavior:

- kernel-based probabilistic classifier
- more flexible than logistic regression
- expensive, so the ablation uses `n_restarts_optimizer=0` to reduce cost

### 7.5 MLP Regressor

**Implemented in:** `ClassicalModels/MLPR.py`  
**Trained by:** `train/train_mlp_regressor.py`

This is scikit-learn's `MLPRegressor` wrapped in a `StandardScaler` pipeline.

Default architecture:

```text
1036 -> 256 -> 128 -> 64 -> 1
```

Layer equation:

```text
h^(l+1) = ReLU(W^(l) h^(l) + b^(l))
```

Implementation details:

- activation = ReLU
- optimizer = Adam
- batch size = 32
- adaptive learning rate
- early stopping enabled
- validation fraction = 0.1

Parameter count:

```text
1036*256 + 256 = 265472
256*128 + 128  = 32896
128*64 + 64    = 8256
64*1 + 1       = 65
Total          = 306689
```

Behavior:

- nonlinear tabular learner
- stronger than linear regression when feature interactions matter
- cheaper than exact GPR

Outputs after running:

- `outputs/classical/mlp_regressor/model.joblib`
- `outputs/classical/mlp_regressor/metrics.json`
- `outputs/classical/mlp_regressor/predictions.csv`

### 7.6 MLP Classifier

**Implemented in:** `ClassicalModels/MLPR.py`  
**Used by:** `ablation/run_feature_ablation.py`

This is the classification counterpart of the scikit-learn MLP regressor.

Behavior:

- same nonlinear feature-learning idea
- outputs probabilities for the binary target

### 7.7 DenseRegressor

**Implemented in:** `DNN/dense_regressor.py`  
**Trained by:** `train/train_dense_regressor.py`

Architecture:

```text
Input(1036) -> Linear -> ReLU -> Dropout
            -> Linear -> ReLU -> Dropout
            -> Linear -> ReLU -> Dropout
            -> Linear(1)
```

Default hidden sizes:

```text
256 -> 128 -> 64
```

Default dropout:

```text
0.15
```

Theory:

- same core math as an MLP
- implemented explicitly in PyTorch for full control over batching, validation, and checkpointing

Parameter count:

```text
306689
```

Training details:

- tabular inputs are standardized with `StandardScaler`
- optimizer = Adam
- loss = MSE
- best validation checkpoint is restored after training

Complexity:

- per epoch is roughly proportional to sample count times parameter count
- far cheaper than exact GPR at this dataset size

Behavior in this project:

- main neural baseline for tabular regression
- dropout helps regularization
- usually the strongest tabular regressor in the current saved results

Outputs after running:

- `outputs/dnn/dense_regressor/model.pt`
- `outputs/dnn/dense_regressor/model_config.json`
- `outputs/dnn/dense_regressor/scaler.joblib`
- `outputs/dnn/dense_regressor/history.csv`
- `outputs/dnn/dense_regressor/metrics.json`
- `outputs/dnn/dense_regressor/predictions.csv`

### 7.8 DenseClassifier

**Implemented in:** `DNN/dense_classifier.py`  
**Used by:** `ablation/run_feature_ablation.py`

This is the binary-classification version of `DenseRegressor`.

Differences from the regressor:

- same architecture
- output is interpreted as a logit
- loss = `BCEWithLogitsLoss`
- probability = `sigmoid(logit)`

Parameter count:

```text
306689
```

### 7.9 GraphCN

**Implemented in:** `GraphML/GraphCN.py`  
**Trained by:** `train/train_graph_cn.py`

This is a simplified GCN-style graph convolution model.

Core equation:

```text
h_v' = sum_{u in N(v) U {v}} (1 / sqrt(d_u d_v)) * W h_u
```

What the code does:

1. adds self-loops
2. linearly projects node features
3. computes degree-normalized weights
4. aggregates neighbor messages
5. mean-pools node embeddings to a graph embedding
6. applies a final linear head for one scalar prediction

Default dimensions:

- input node features = 3
- hidden channels = 64

Parameter count:

```text
321
```

Complexity per forward pass:

- projection: `O(|V| * F_in * F_out)`
- message passing: `O(|E| * F_out)`
- pooling: `O(|V| * F_out)`

Behavior:

- structure-aware baseline with degree normalization
- shallow, so it mainly captures local structure

Outputs after running:

- `outputs/graphml/graph_cn/model.pt`
- `outputs/graphml/graph_cn/history.csv`
- `outputs/graphml/graph_cn/metrics.json`
- `outputs/graphml/graph_cn/predictions.csv`

### 7.10 GraphNET

**Implemented in:** `GraphML/GraphNET.py`  
**Trained by:** `train/train_graph_net.py`

This is an attention-style graph message-passing model.

Message equations:

```text
e_uv     = LeakyReLU(a^T [W x_u || W x_v])
alpha_uv = softmax_u(e_uv)
m_v      = sum_u alpha_uv * W x_u
```

Meaning:

- neighbors do not all contribute equally
- the network learns attention weights over neighbors

Parameter count:

```text
385
```

Complexity per forward pass:

- projection: `O(|V| * F_in * F_out)`
- attention/message computation: `O(|E| * F_out)`
- pooling and head: `O(|V| * F_out + F_out)`

Behavior:

- more adaptive than plain convolution
- can emphasize more informative neighbors
- still shallow and local

Implementation note:

- this model does not add self-loops
- self information affects the attention score through `x_i`, but the propagated value is still neighbor-weighted `x_j`

### 7.11 GraphSAGE

**Implemented in:** `GraphML/GraphSAGE.py`  
**Trained by:** `train/train_graph_sage.py`

This is a simplified GraphSAGE-style model with mean aggregation.

Concept:

```text
m_v = mean_{u in N(v)} x_u
z_v = [x_v || m_v]
```

Important code-specific detail:

- this implementation performs `global_mean_pool` **before** the main projection layer
- canonical GraphSAGE usually transforms node representations before pooling

So the project behavior is effectively:

```text
z_G   = mean_{v in G} [x_v || mean_{u in N(v)} x_u]
y_hat = W2 (W1 z_G + b1) + b2
```

Parameter count:

```text
513
```

Complexity per forward pass:

- aggregation: `O(|E| * F_in)`
- pooling: `O(|V| * F_in)`
- graph-level head: small, because it acts after pooling

Behavior:

- keeps self information by concatenating original node features with aggregated neighbor features
- lightweight
- cheaper but less expressive than a deeper node-level GraphSAGE stack

### 7.12 GraphMP

**Implemented in:** `GraphML/GraphMP.py`  
**Trained by:** `train/train_graph_mp.py`

This is a custom message-passing network.

Message equation:

```text
m_{u->v} = W_m [x_v || x_u] + b_m
```

Aggregation and graph update:

```text
m_v   = sum_{u in N(v)} m_{u->v}
h_G   = mean_{v in G} m_v
z_G   = W_u h_G + b_u
y_hat = W_o z_G + b_o
```

Parameter count:

```text
4673
```

This is the largest graph model in the repository because it learns a richer message function.

Complexity per forward pass:

- message creation: `O(|E| * 2F_in * F_out)`
- pooling: `O(|V| * F_out)`
- graph-level update: `O(F_out^2)`

Behavior:

- most expressive graph architecture in the current repository
- uses both sender and receiver features inside each message
- still ignores `edge_attr`, so bond weights do not affect learned messages

### 7.13 Graph vs tabular learning summary

Tabular models learn from precomputed feature vectors:

- easier to train
- often strong with good fingerprints and descriptors
- do not preserve explicit connectivity directly

Graph models learn from molecular structure:

- more natural for chemistry
- can reason over atoms and neighborhoods
- usually cost more per epoch than linear baselines

## 8. Complexity Summary

| Model | Default params | Training complexity | Main practical effect |
|---|---:|---|---|
| Linear Regression | 1037 | about `O(n d^2)` | very fast baseline |
| Logistic Regression | 1037 | about `O(T n d)` | fast classifier baseline |
| GPR / GPC | not parameter-count driven | `O(n^3)` | expensive kernel method |
| MLPRegressor / Classifier | 306689 | epoch-based dense backprop | nonlinear tabular learner |
| DenseRegressor / Classifier | 306689 | epoch-based dense backprop | custom PyTorch tabular learner |
| GraphCN | 321 | about linear in nodes and edges | normalized graph convolution |
| GraphNET | 385 | about linear in nodes and edges | attention-weighted messages |
| GraphSAGE | 513 | about linear in nodes and edges | mean neighbor aggregation |
| GraphMP | 4673 | about linear in edges plus graph head | richer custom message function |

The key lesson is that parameter count does not fully explain runtime:

- GPR has few optimized hyperparameters but is slow because of dense kernel algebra.
- DNNs have many trainable parameters but remain tractable because they use mini-batch optimization.

## 9. Train Scripts And Outputs

| Script | Model | Output folder |
|---|---|---|
| `train/train_linear_regression.py` | Linear regression | `outputs/classical/linear_regression/` |
| `train/train_gaussian_process.py` | Gaussian process regression | `outputs/classical/gaussian_process/` |
| `train/train_mlp_regressor.py` | scikit-learn MLP regressor | `outputs/classical/mlp_regressor/` |
| `train/train_dense_regressor.py` | PyTorch dense regressor | `outputs/dnn/dense_regressor/` |
| `train/train_graph_cn.py` | GraphCN | `outputs/graphml/graph_cn/` |
| `train/train_graph_net.py` | GraphNET | `outputs/graphml/graph_net/` |
| `train/train_graph_sage.py` | GraphSAGE | `outputs/graphml/graph_sage/` |
| `train/train_graph_mp.py` | GraphMP | `outputs/graphml/graph_mp/` |
| `train/run_full_pipeline.py` | runs all regression scripts sequentially | all folders above |

Practical note:

- `run_full_pipeline.py` also runs GPR, so the full pipeline can be much slower than the individual non-GP scripts.

## 10. Hyperparameter Tuning Scripts

### 10.1 `tuning/tune_classical_models.py`

This script compares:

- linear regression baseline
- randomized search over Gaussian process regression
- randomized search over MLP regressor

Search details:

- GPR: `6` random configurations, `3`-fold CV
- MLP: `6` random configurations, `3`-fold CV

Saved output:

- `outputs/tuning/classical_tuning_results.csv`

### 10.2 `tuning/tune_dnn.py`

This script tries three dense configurations:

| Hidden layers | Dropout | Learning rate |
|---|---:|---:|
| `(128, 64)` | 0.10 | 1e-3 |
| `(256, 128, 64)` | 0.15 | 1e-3 |
| `(512, 256, 128)` | 0.20 | 5e-4 |

Each config trains for:

- `80` epochs
- batch size `64`

Saved outputs:

- `outputs/tuning/dnn_tuning_results.csv`
- `outputs/tuning/best_dense_scaler.joblib`
- `outputs/tuning/best_dense_config.json`

Important detail:

- this tuning script saves the best config and scaler
- it does **not** save the tuned dense model weights

### 10.3 `tuning/tune_graph_models.py`

This script tunes:

- `graph_cn`
- `graph_net`
- `graph_sage`

It does **not** tune `graph_mp`.

Search space:

| Hidden channels | Learning rate |
|---:|---:|
| 32 | 1e-3 |
| 64 | 1e-3 |
| 128 | 5e-4 |

Each run trains for:

- `60` epochs
- batch size `32`

Total runs:

```text
3 models x 3 configs = 9 graph runs
```

Saved output:

- `outputs/tuning/graph_tuning_results.csv`

## 11. Ablation Scripts

The ablation studies ask what changes when solubility prediction is simplified into a binary decision.

Threshold rule:

```text
positive if Solubility >= -3.0
negative otherwise
```

### 11.1 `ablation/run_feature_ablation.py`

This compares tabular classification across:

- fingerprint only
- descriptor only
- combined features

and across four models:

- logistic regression
- Gaussian process classifier
- MLP classifier
- dense classifier

Total runs:

```text
3 feature modes x 4 models = 12 experiments
```

Saved output:

- `outputs/ablation/tabular_classification_ablation.csv`

### 11.2 `ablation/run_graph_ablation.py`

This compares graph classification across:

- `atomic_number`
- `atomic_number_degree`
- `full`

and across four graph models:

- GraphCN
- GraphNET
- GraphSAGE
- GraphMP

Total runs:

```text
3 feature variants x 4 graph models = 12 experiments
```

Saved output:

- `outputs/ablation/graph_classification_ablation.csv`

## 12. Streamlit Dashboard

`app.py` provides a lightweight UI with three tabs:

- `Dataset`
- `Model Results`
- `Studies`

It does three useful things:

1. shows the cleaned reference dataset
2. builds a leaderboard from every `metrics.json` under `outputs/*/*/`
3. lets you inspect prediction CSVs and study tables

This is useful for presentations because the experiment outputs become browsable without writing extra analysis code.

## 13. Current Results Already Present In This Workspace

At the moment this workspace contains saved outputs for:

- linear regression
- Gaussian process regression
- dense regressor

It does **not** currently contain saved outputs for:

- `mlp_regressor`
- graph regressors
- tuning runs
- ablation runs

### 13.1 Current regression leaderboard

| Rank | Model | RMSE | MAE | R^2 |
|---:|---|---:|---:|---:|
| 1 | DenseRegressor | 1.2306 | 0.8730 | 0.7207 |
| 2 | LinearRegression | 1.5715 | 1.1914 | 0.5445 |
| 3 | GaussianProcessRegressor | 1.6700 | 1.1510 | 0.4856 |

Interpretation:

- the current best saved model is the PyTorch dense regressor
- linear regression is a respectable simple baseline
- GPR is the most expensive saved model here but not the best performer on this run

### 13.2 Dense regressor training behavior

From `outputs/dnn/dense_regressor/history.csv`:

- trained for `150` epochs
- best validation loss occurred at **epoch 134**
- best validation loss was about **1.5817**

This matches the code behavior of restoring the best validation checkpoint rather than using the final epoch blindly.

### 13.3 Error-analysis view from the saved predictions

| Model | Mean absolute error | Median absolute error | Max absolute error |
|---|---:|---:|---:|
| LinearRegression | 1.1914 | 0.9678 | 13.9064 |
| GaussianProcessRegressor | 1.1510 | 0.7581 | 7.9348 |
| DenseRegressor | 0.8730 | 0.6147 | 6.6744 |

This again shows the dense regressor has the strongest current test-set behavior among the saved runs.

## 14. What You Get After Running Each Category

### 14.1 Data-preparation scripts

Running:

```powershell
python utils\make_data.py
python utils\make_graph_data.py
```

produces:

- `data/X.npy`
- `data/y.npy`
- `data/aqsoldb_graph_dataset.pt`

These are cached representations of the cleaned dataset.

### 14.2 Regression training scripts

Running any script from `train/` produces:

- a saved model
- a metrics JSON
- a prediction CSV
- and for neural models, a history CSV

Meaning of those results:

- `metrics.json` tells you how the model performed on unseen test data
- `predictions.csv` lets you inspect molecule-level errors
- model weights let you reuse or analyze the trained model later

### 14.3 Tuning scripts

Running scripts from `tuning/` produces:

- ranked comparison tables for hyperparameter settings
- best-config metadata for the dense model

These outputs answer:

- which hyperparameters worked best
- how sensitive each family is to architecture or optimization choices

### 14.4 Ablation scripts

Running scripts from `ablation/` produces:

- binary-classification comparison tables
- feature-sensitivity or node-feature-sensitivity results

These outputs answer:

- what changes when regression becomes classification
- whether fingerprints, descriptors, or node features matter most
- which model family is most robust under simplified labels

## 15. Important Implementation Notes And Caveats

These details explain the code behavior, not just the textbook theory.

1. The project recomputes tabular descriptors from SMILES with RDKit even though the CSV already contains several descriptor columns.
2. Cached arrays and graph datasets are generated, but the training scripts rebuild features and graphs instead of loading the caches.
3. Graph `edge_attr` is created from bond type but ignored by every current graph model.
4. All graph models are shallow, single-message-passing architectures with graph-level pooling and a simple output head.
5. `GraphSAGE` and `GraphMP` apply important learned transformations after graph pooling, which makes them simpler than more standard multi-layer graph formulations.
6. The dense neural loops save the best validation checkpoint, but they still run through all epochs.
7. Full-pipeline runtime is strongly affected by Gaussian process training.
8. Graph tuning currently excludes `GraphMP`, so not every graph architecture is tuned.

## 16. Big Picture Takeaway

This project is a controlled comparison of **representation**, **algorithm**, and **computational tradeoff**.

The same molecule is viewed in three ways:

- as a handcrafted tabular vector
- as a nonlinear tabular input to a dense neural network
- as a graph of atoms and bonds

The same target is viewed in two ways:

- as a continuous regression value
- as a thresholded binary label

The strongest mental model for the whole repository is:

```text
SMILES -> representation -> split -> train -> validate -> test -> save -> compare
```

That one line is the entire project workflow.

# AqSolDB Course Project

This version of the repository is intentionally simple.

The goal is to show:

- classical regression models on molecular features
- deep neural networks on tabular molecular features
- graph neural networks on molecular graphs
- a clear end-to-end pipeline from data loading to saved results
- hyperparameter tuning and ablation studies in separate folders

The main training pipeline is regression.

Torch and PyTorch Geometric models use GPU automatically when CUDA is available. Scikit-learn models in this repository remain CPU-only unless the project is rewritten around a CUDA-backed library.

The ablation folder is used for a separate binary classification study, where the continuous solubility target is thresholded into two classes.

## Project Structure

```text
AqSolDB/
|-- ClassicalModels/   # builders for regression baselines and their classification-study variants
|-- GraphML/           # graph neural network model definitions
|-- DNN/               # dense neural network model definitions
|-- EDA/               # exploratory analysis scripts for report visuals
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
python train\train_dense_regressor.py --device auto
python train\train_mlp_regressor.py
```

Graph models:

```powershell
python train\train_graph_cn.py --device auto
python train\train_graph_net.py --device auto
python train\train_graph_sage.py --device auto
python train\train_graph_mp.py --device auto
```

Run the whole training pipeline:

```powershell
python train\run_full_pipeline.py --device auto
```

### 3. Run hyperparameter tuning

```powershell
python tuning\tune_classical_models.py
python tuning\tune_dnn.py --device auto
python tuning\tune_graph_models.py --device auto
```

### 4. Run ablation studies

These scripts run binary classification experiments, not regression.
By default they label a molecule as positive when `Solubility >= -3.0`.

```powershell
python ablation\run_feature_ablation.py --device auto
python ablation\run_graph_ablation.py --device auto
```

### 5. Inspect saved outputs

```powershell
streamlit run app.py
```

### 6. Generate EDA figures for the report

```powershell
python EDA\generate_eda_report.py
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
|   |-- graph_mp_tuned/
|-- tuning/
|-- ablation/
```

Each model folder stores:

- `metrics.json`
- `predictions.csv`
- saved model weights
- training history for neural models

## Current Graph Setup

Graph models are trained on molecular graphs built from RDKit in `utils/data_utils.py`.

Current default feature dimensions:

- node feature vector size = `11`
- edge feature vector size = `10`
- global per-graph descriptor vector size = `12`

The `11` node features are:

- atomic number
- degree
- formal charge
- total hydrogens
- implicit valence
- total valence
- aromatic flag
- ring flag
- `sp`
- `sp2`
- `sp3`

The `10` edge features are:

- bond order as a float
- single-bond flag
- double-bond flag
- triple-bond flag
- aromatic-bond flag
- conjugation flag
- ring-bond flag
- cis / Z stereo flag
- trans / E stereo flag
- stereo-any flag

The `12` global graph descriptors are:

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

Graph feature usage by model:

- `GraphCN` uses node features plus the `12` global descriptors after pooling.
- `GraphSAGE` uses node features plus the `12` global descriptors after pooling.
- `GraphNET` uses node features, `10`-D edge features, and the `12` global descriptors.
- `GraphMP` uses node features, `10`-D edge features, and the `12` global descriptors.

## Graph Architecture Summary

### GraphCN

- 2 `GCNConv` layers
- 2 batch-normalization layers
- global mean pooling
- MLP readout on `[graph_embedding || global_descriptors]`

### GraphNET

- node encoder: `Linear(11 -> H)`
- edge encoder: `Linear(10 -> H)`, `Linear(H -> H)`
- 3 `GINEConv` message-passing blocks with residual connections
- mean, max, and add global pooling
- global descriptor encoder: `Linear(12 -> H)`
- MLP readout on pooled graph features plus encoded descriptors

### GraphSAGE

- 2 `SAGEConv` layers
- 2 batch-normalization layers
- global mean pooling
- MLP readout on `[graph_embedding || global_descriptors]`

### GraphMP

- 2 `NNConv` layers
- edge-conditioned message networks driven by the `10`-D bond feature vector
- 2 batch-normalization layers
- global mean pooling
- MLP readout on `[graph_embedding || global_descriptors]`

## Trainable Parameter Counts

The counts below are for the current default graph training setup:

- hidden channels `H = 64`
- node dimension `F_node = 11`
- edge dimension `F_edge = 10`
- global descriptor dimension `F_global = 12`

Current trainable parameter counts:

| Model | Trainable parameters |
|---|---:|
| `GraphCN` | `7,681` |
| `GraphNET` | `54,465` |
| `GraphSAGE` | `12,481` |
| `GraphMP` | `321,089` |

How the count is calculated:

- for a linear layer `Linear(a, b)`, parameters = `a * b + b`
- for `BatchNorm1d(b)`, trainable parameters = `2 * b` for scale and shift
- total trainable parameters = sum of `p.numel()` over all parameters where `p.requires_grad`

Examples:

- `GraphCN`:
  `GCNConv(11,64) + GCNConv(64,64) + 2 x BatchNorm1d(64) + Linear(76,32) + Linear(32,1) = 7,681`
- `GraphNET`:
  node encoder + edge encoder + `3 x GINEConv` MLPs + `3 x BatchNorm1d(64)` + global encoder + readout MLP = `54,465`
- `GraphSAGE`:
  `SAGEConv(11,64) + SAGEConv(64,64) + 2 x BatchNorm1d(64) + Linear(76,32) + Linear(32,1) = 12,481`
- `GraphMP`:
  two edge networks for `NNConv` + root projections + `2 x BatchNorm1d(64)` + readout MLP = `321,089`

## Best CUDA GraphMP Result

Manual CUDA tuning on `node1` found a stronger `GraphMP` configuration than the default training script:

- hidden channels = `128`
- dropout = `0.05`
- learning rate = `0.0005`
- weight decay = `1e-5`
- batch size = `48`
- max epochs = `180`
- early stop epoch = `101`
- test `R2 = 0.7915`

This tuned run uses the same `11`-D node features, `10`-D edge features, and `12`-D global descriptors, but with a larger hidden width than the default `64`.

## Notes

- `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, and `GaussianProcessRegressor` are the main classical regression baselines.
- `DenseRegressor` and scikit-learn `MLPRegressor` live under `DNN/` as the tabular neural-network baselines.
- `GraphCN`, `GraphNET`, `GraphSAGE`, and `GraphMP` are graph-level regression models in the main pipeline.
- `GraphNET` and `GraphMP` explicitly use the bond-level `edge_attr` tensor produced by `build_graph_dataset()`.
- The ablation scripts convert the problem into binary classification for all families so you can compare tabular and graph methods on the same discrete target.
- The project favors readability over packaging complexity so it is easier to explain in a course report or viva.

## Study Notes (Deep Dive)

The sections below are a deeper, more theory-oriented walkthrough of the same project — useful for a course report or viva prep. They cover the same repository described above, just in more mathematical and behavioral detail (parameter counts, complexity, per-model equations, and Q&A prompts).

### 1. What This Project Is Doing

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

### 2. Repository Map

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

### 3. End-To-End Flow

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

### 4. Data Cleaning And Representation

#### 4.1 Cleaned dataset

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

#### 4.2 Tabular representation

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

#### 4.3 Graph representation

For graph models, each molecule becomes a graph:

- nodes = atoms
- edges = bonds
- graph label = molecule solubility

Default node features:

- atomic number
- atom degree
- formal charge
- total hydrogens
- implicit valence
- total valence
- aromaticity flag
- ring-membership flag
- `sp`
- `sp2`
- `sp3`

Graph feature variants used in ablation:

| Variant | Node feature dimension |
|---|---:|
| `atomic_number` | 1 |
| `atomic_number_degree` | 2 |
| `full` | 11 |

Default bond features are encoded as a `10`-dimensional edge attribute vector:

- bond order as a float
- single-bond flag
- double-bond flag
- triple-bond flag
- aromatic-bond flag
- conjugation flag
- ring-bond flag
- cis / Z stereo flag
- trans / E stereo flag
- stereo-any flag

Important code fact:

- `build_graph_dataset()` stores `edge_attr`
- `GraphNET` and `GraphMP` use `edge_attr`
- `GraphCN` and `GraphSAGE` ignore `edge_attr`

Each graph also stores a `12`-dimensional global descriptor vector built from:

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

Cached graph dataset statistics from `data/aqsoldb_graph_dataset.pt`:

| Statistic | Value |
|---|---:|
| Graphs | 9980 |
| Node feature dimension | 11 |
| Edge feature dimension | 10 |
| Global descriptor dimension | 12 |
| Average atoms per molecule | 17.3830 |
| Median atoms per molecule | 15 |
| Average directed edges per molecule | 35.2669 |
| Median directed edges per molecule | 30 |
| Minimum atoms | 1 |
| Maximum atoms | 388 |
| Graphs with zero bonds | 149 |

### 5. Splits, Losses, Metrics, And Artifacts

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

### 6. Shared Training Logic

Reusable training logic lives in `utils/training_utils.py`.

#### 6.1 Reproducibility

`set_global_seed()` seeds:

- Python `random`
- NumPy
- PyTorch CPU and CUDA

#### 6.2 Neural training loop behavior

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

#### 6.3 Graph-level pooling

All graph models perform graph-level prediction using mean pooling:

```text
h_G = (1 / |V|) * sum_{v in G} h_v
```

This is `global_mean_pool`, which gives one vector per molecule before the final prediction head.

### 7. Algorithms In The Project

#### 7.1 Linear Regression

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

#### 7.2 Logistic Regression

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

#### 7.3 Gaussian Process Regression

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

#### 7.4 Gaussian Process Classifier

**Implemented in:** `ClassicalModels/GPR.py`  
**Used by:** `ablation/run_feature_ablation.py`

This is the binary-classification version of the GP idea.

Behavior:

- kernel-based probabilistic classifier
- more flexible than logistic regression
- expensive, so the ablation uses `n_restarts_optimizer=0` to reduce cost

#### 7.5 MLP Regressor

**Implemented in:** `DNN/sklearn_mlp.py`  
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

- `outputs/dnn/sklearn_mlp_regressor/model.joblib`
- `outputs/dnn/sklearn_mlp_regressor/metrics.json`
- `outputs/dnn/sklearn_mlp_regressor/predictions.csv`

#### 7.6 MLP Classifier

**Implemented in:** `DNN/sklearn_mlp.py`  
**Used by:** `ablation/run_feature_ablation.py`

This is the classification counterpart of the scikit-learn MLP regressor.

Behavior:

- same nonlinear feature-learning idea
- outputs probabilities for the binary target

#### 7.7 DenseRegressor

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

#### 7.8 DenseClassifier

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

#### 7.9 GraphCN

**Implemented in:** `GraphML/GraphCN.py`  
**Trained by:** `train/train_graph_cn.py`

This is a 2-layer GCN-style graph convolution model with graph-level descriptor fusion.

Current default dimensions:

- node feature dimension = `11`
- global descriptor dimension = `12`
- hidden channels = `64`

Forward structure:

```text
x
-> GCNConv(11, 64)
-> BatchNorm1d(64)
-> ReLU
-> GCNConv(64, 64)
-> BatchNorm1d(64)
-> ReLU
-> global_mean_pool
-> concat with 12-D global descriptor vector
-> Linear(76, 32)
-> ReLU
-> Linear(32, 1)
```

Trainable parameter count:

```text
7,681
```

Calculation:

```text
GCNConv(11,64)      = 11*64 + 64     =    768
GCNConv(64,64)      = 64*64 + 64     =  4,160
2 x BatchNorm1d(64) = 2*(64 + 64)    =    256
Linear(76,32)       = 76*32 + 32     =  2,464
Linear(32,1)        = 32*1 + 1       =     33
Total                                   7,681
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

#### 7.10 GraphNET

**Implemented in:** `GraphML/GraphNET.py`  
**Trained by:** `train/train_graph_net.py`

This is an edge-aware residual graph message-passing model.

The current implementation is no longer GAT-based. It is an edge-aware residual `GINEConv` model with explicit encoders for node, edge, and global molecular descriptors.

Current default dimensions:

- node feature dimension = `11`
- edge feature dimension = `10`
- global descriptor dimension = `12`
- hidden channels = `64`

Forward structure:

```text
node encoder:   Linear(11,64)
edge encoder:   Linear(10,64) -> Linear(64,64)
message stack:  3 x GINEConv(MLP(64 -> 64 -> 64)) with residual connections
pooling:        global_mean_pool, global_max_pool, global_add_pool
global branch:  Linear(12,64)
readout:        Linear(320,64) -> Linear(64,32) -> Linear(32,1)
```

The readout input has size `320` because it concatenates:

- mean pooled graph embedding = `64`
- max pooled graph embedding = `64`
- add pooled graph embedding = `64`
- encoded global descriptor vector = `64`
- pooled mean minus pooled max = `64`

Trainable parameter count:

```text
54,465
```

Calculation:

```text
node encoder           = 11*64 + 64                    =    768
edge encoder           = (10*64 + 64) + (64*64 + 64)  =  4,864
3 x GINE MLP blocks    = 3 * ((64*64 + 64) * 2)       = 24,960
3 x BatchNorm1d(64)    = 3 * (64 + 64)                =    384
global encoder         = 12*64 + 64                   =    832
Linear(320,64)         = 320*64 + 64                  = 20,544
Linear(64,32)          = 64*32 + 32                   =  2,080
Linear(32,1)           = 32*1 + 1                     =     33
Total                                                   54,465
```

Complexity per forward pass:

- node encoding: `O(|V| * F_node * H)`
- edge encoding: `O(|E| * F_edge * H)`
- message passing: `O(3 * |E| * H^2)` inside the `GINE` MLPs
- pooling and head: `O(|V| * H + H^2)`

Behavior:

- explicitly uses bond-level edge features
- mixes three graph-level pooling operators instead of only mean pooling
- stronger graph head than `GraphCN`
- more expensive than `GraphCN` and `GraphSAGE`, but much smaller than `GraphMP`

#### 7.11 GraphSAGE

**Implemented in:** `GraphML/GraphSAGE.py`  
**Trained by:** `train/train_graph_sage.py`

This is a 2-layer GraphSAGE-style model with mean aggregation and graph-level descriptor fusion.

Current default dimensions:

- node feature dimension = `11`
- global descriptor dimension = `12`
- hidden channels = `64`

Forward structure:

```text
x
-> SAGEConv(11, 64)
-> BatchNorm1d(64)
-> ReLU
-> SAGEConv(64, 64)
-> BatchNorm1d(64)
-> ReLU
-> global_mean_pool
-> concat with 12-D global descriptor vector
-> Linear(76, 32)
-> ReLU
-> Linear(32, 1)
```

Trainable parameter count:

```text
12,481
```

Calculation:

```text
SAGEConv(11,64)      = 2*(11*64) + 64        =  1,472
SAGEConv(64,64)      = 2*(64*64) + 64        =  8,256
2 x BatchNorm1d(64)  = 2*(64 + 64)           =    256
Linear(76,32)        = 76*32 + 32            =  2,464
Linear(32,1)         = 32*1 + 1              =     33
Total                                          12,481
```

Complexity per forward pass:

- aggregation: `O(|E| * H)`
- pooling: `O(|V| * H)`
- graph-level head: `O(H * (H + F_global))`

Behavior:

- keeps self and neighbor information through the GraphSAGE update
- still ignores edge features
- lightweight compared with `GraphNET` and `GraphMP`

#### 7.12 GraphMP

**Implemented in:** `GraphML/GraphMP.py`  
**Trained by:** `train/train_graph_mp.py`

This is an edge-conditioned message-passing network built with `NNConv`.

Current default dimensions:

- node feature dimension = `11`
- edge feature dimension = `10`
- global descriptor dimension = `12`
- hidden channels = `64`

Forward structure:

```text
edge MLP 1:  Linear(10,64) -> Linear(64,11*64)
edge MLP 2:  Linear(10,64) -> Linear(64,64*64)
x
-> NNConv(11,64, edge_mlp_1)
-> BatchNorm1d(64)
-> ReLU
-> NNConv(64,64, edge_mlp_2)
-> BatchNorm1d(64)
-> ReLU
-> global_mean_pool
-> concat with 12-D global descriptor vector
-> Linear(76, 32)
-> ReLU
-> Linear(32, 1)
```

Trainable parameter count:

```text
321,089
```

Calculation:

```text
edge MLP 1           = (10*64 + 64) + (64*(11*64) + 11*64) =  46,464
edge MLP 2           = (10*64 + 64) + (64*(64*64) + 64*64) = 266,944
NNConv root 1        = 11*64 + 64                          =     768
NNConv root 2        = 64*64 + 64                          =   4,160
2 x BatchNorm1d(64)  = 2*(64 + 64)                         =     256
Linear(76,32)        = 76*32 + 32                          =   2,464
Linear(32,1)         = 32*1 + 1                             =      33
Total                                                        321,089
```

This is the largest graph model in the repository at the default hidden size because the `NNConv` edge networks generate full edge-conditioned weight matrices.

Complexity per forward pass:

- edge-network evaluation: `O(|E| * (F_edge * H + H^2))`
- message passing: `O(|E| * H^2)`
- pooling: `O(|V| * H)`
- graph-level head: `O(H * (H + F_global))`

Behavior:

- most expressive graph architecture in the current repository
- explicitly uses the `10`-D bond feature vector
- strongest graph result so far came from manual CUDA tuning on `node1`

Best tuned CUDA result so far:

```text
hidden_channels = 128
dropout         = 0.05
learning_rate   = 5e-4
weight_decay    = 1e-5
batch_size      = 48
epochs          = 180
early stop      = 101
test R2         = 0.7915
```

#### 7.13 Graph vs tabular learning summary

Tabular models learn from precomputed feature vectors:

- easier to train
- often strong with good fingerprints and descriptors
- do not preserve explicit connectivity directly

Graph models learn from molecular structure:

- more natural for chemistry
- can reason over atoms and neighborhoods
- usually cost more per epoch than linear baselines

### 8. Complexity Summary

| Model | Default params | Training complexity | Main practical effect |
|---|---:|---|---|
| Linear Regression | 1037 | about `O(n d^2)` | very fast baseline |
| Logistic Regression | 1037 | about `O(T n d)` | fast classifier baseline |
| GPR / GPC | not parameter-count driven | `O(n^3)` | expensive kernel method |
| MLPRegressor / Classifier | 306689 | epoch-based dense backprop | nonlinear tabular learner |
| DenseRegressor / Classifier | 306689 | epoch-based dense backprop | custom PyTorch tabular learner |
| GraphCN | 7,681 | about linear in nodes and edges | normalized graph convolution with global descriptor fusion |
| GraphNET | 54,465 | about linear in nodes and edges plus edge MLPs | edge-aware residual GINE with multi-pooling readout |
| GraphSAGE | 12,481 | about linear in nodes and edges | mean neighbor aggregation with global descriptor fusion |
| GraphMP | 321,089 | about linear in edges plus graph head | edge-conditioned `NNConv` with learned message kernels |

The key lesson is that parameter count does not fully explain runtime:

- GPR has few optimized hyperparameters but is slow because of dense kernel algebra.
- DNNs have many trainable parameters but remain tractable because they use mini-batch optimization.

### 9. Train Scripts And Outputs

| Script | Model | Output folder |
|---|---|---|
| `train/train_linear_regression.py` | Linear regression | `outputs/classical/linear_regression/` |
| `train/train_gaussian_process.py` | Gaussian process regression | `outputs/classical/gaussian_process/` |
| `train/train_mlp_regressor.py` | scikit-learn MLP regressor | `outputs/dnn/sklearn_mlp_regressor/` |
| `train/train_dense_regressor.py` | PyTorch dense regressor | `outputs/dnn/dense_regressor/` |
| `train/train_graph_cn.py` | GraphCN | `outputs/graphml/graph_cn/` |
| `train/train_graph_net.py` | GraphNET | `outputs/graphml/graph_net/` |
| `train/train_graph_sage.py` | GraphSAGE | `outputs/graphml/graph_sage/` |
| `train/train_graph_mp.py` | GraphMP | `outputs/graphml/graph_mp/` |
| manual CUDA rerun | tuned GraphMP | `outputs/graphml/graph_mp_tuned/` |
| `train/run_full_pipeline.py` | runs all regression scripts sequentially | all folders above |

Practical note:

- `run_full_pipeline.py` also runs GPR, so the full pipeline can be much slower than the individual non-GP scripts.

### 10. Hyperparameter Tuning Scripts

#### 10.1 `tuning/tune_classical_models.py`

This script compares:

- linear regression baseline
- randomized search over Gaussian process regression
- randomized search over MLP regressor

Search details:

- GPR: `6` random configurations, `3`-fold CV
- MLP: `6` random configurations, `3`-fold CV

Saved output:

- `outputs/tuning/classical_tuning_results.csv`

#### 10.2 `tuning/tune_dnn.py`

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

#### 10.3 `tuning/tune_graph_models.py`

This script tunes:

- `graph_cn`
- `graph_net`
- `graph_sage`

It does **not** tune `graph_mp`.

Manual follow-up tuning on `node1` with CUDA found a stronger `GraphMP` configuration:

- hidden channels = `128`
- dropout = `0.05`
- learning rate = `5e-4`
- weight decay = `1e-5`
- batch size = `48`
- test `R2 = 0.7915`

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

### 11. Ablation Scripts

The ablation studies ask what changes when solubility prediction is simplified into a binary decision.

Threshold rule:

```text
positive if Solubility >= -3.0
negative otherwise
```

#### 11.1 `ablation/run_feature_ablation.py`

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

#### 11.2 `ablation/run_graph_ablation.py`

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

### 12. Streamlit Dashboard

`app.py` provides a lightweight UI with three tabs:

- `Dataset`
- `Model Results`
- `Studies`

It does three useful things:

1. shows the cleaned reference dataset
2. builds a leaderboard from every `metrics.json` under `outputs/*/*/`
3. lets you inspect prediction CSVs and study tables

This is useful for presentations because the experiment outputs become browsable without writing extra analysis code.

### 13. Current Results Already Present In This Workspace

This workspace contains saved outputs for every model family: classical
regression, DNN, and all four graph models (including the manually tuned
`graph_mp_tuned`), plus PCA3D-feature variants. A git-tracked snapshot of
these results — a combined leaderboard plus per-model metrics and sample
predictions — lives in [`results/`](results/) so they're visible on GitHub
without re-running the pipeline; full predictions, model weights, and
training histories stay in the local (gitignored) `outputs/` folder.

#### 13.1 Current regression leaderboard

| Rank | Model | RMSE | MAE | R^2 |
|---:|---|---:|---:|---:|
| 1 | GraphMP (tuned) | 1.0471 | 0.7435 | 0.7978 |
| 2 | GraphSAGE | 1.0766 | 0.7701 | 0.7862 |
| 3 | GraphNET | 1.0992 | 0.7838 | 0.7771 |
| 4 | GraphMP | 1.1178 | 0.7870 | 0.7695 |
| 5 | MLPRegressor (sklearn) | 1.1618 | 0.8117 | 0.7512 |
| 6 | DenseRegressor | 1.1771 | 0.8473 | 0.7446 |
| 7 | GraphCN | 1.1943 | 0.8670 | 0.7369 |
| 8 | MLPRegressor (classical) | 1.2088 | 0.8517 | 0.7305 |
| 9 | GaussianProcessRegressor | 1.2558 | 0.8936 | 0.7091 |
| 10 | LassoRegression | 1.3714 | 1.0245 | 0.6531 |
| 11 | RidgeRegression | 1.4308 | 1.0396 | 0.6224 |
| 12 | LinearRegression | 1.4616 | 1.0629 | 0.6062 |

Full table (including PCA3D variants) is in `results/leaderboard.csv`.

Interpretation:

- all four graph models now outperform every tabular model, with the
  manually tuned `GraphMP` in the lead — edge-aware message passing plus
  the global descriptor fusion pays off once properly tuned
- the DNN and scikit-learn MLP remain the strongest tabular learners
- GPR is still the most expensive model to train here, and it is now
  solidly mid-pack rather than the weakest result

#### 13.2 Dense regressor training behavior

From `outputs/dnn/dense_regressor/history.csv`:

- trained for `105` epochs (of a `150`-epoch budget)
- best validation loss occurred at **epoch 85**
- best validation loss was about **0.2398**

This matches the code behavior of restoring the best validation checkpoint rather than using the final epoch blindly.

#### 13.3 Error-analysis view from the saved predictions

| Model | Mean absolute error | Median absolute error | Max absolute error |
|---|---:|---:|---:|
| LinearRegression | 1.0629 | 0.8316 | 15.6072 |
| GaussianProcessRegressor | 0.8936 | 0.6269 | 6.0456 |
| DenseRegressor | 0.8473 | 0.6076 | 6.3570 |

Among these three, the dense regressor has the best mean/median error, though GPR edges it out on the single worst-case prediction — but the graph models in 13.1 beat all three on RMSE/R².

### 14. What You Get After Running Each Category

#### 14.1 Data-preparation scripts

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

#### 14.2 Regression training scripts

Running any script from `train/` produces:

- a saved model
- a metrics JSON
- a prediction CSV
- and for neural models, a history CSV

Meaning of those results:

- `metrics.json` tells you how the model performed on unseen test data
- `predictions.csv` lets you inspect molecule-level errors
- model weights let you reuse or analyze the trained model later

#### 14.3 Tuning scripts

Running scripts from `tuning/` produces:

- ranked comparison tables for hyperparameter settings
- best-config metadata for the dense model

These outputs answer:

- which hyperparameters worked best
- how sensitive each family is to architecture or optimization choices

#### 14.4 Ablation scripts

Running scripts from `ablation/` produces:

- binary-classification comparison tables
- feature-sensitivity or node-feature-sensitivity results

These outputs answer:

- what changes when regression becomes classification
- whether fingerprints, descriptors, or node features matter most
- which model family is most robust under simplified labels

### 15. Important Implementation Notes And Caveats

These details explain the code behavior, not just the textbook theory.

1. The project recomputes tabular descriptors from SMILES with RDKit even though the CSV already contains several descriptor columns.
2. Cached arrays and graph datasets are generated, but the training scripts rebuild features and graphs instead of loading the caches.
3. Graph `edge_attr` is explicitly used by `GraphNET` and `GraphMP`, but ignored by `GraphCN` and `GraphSAGE`.
4. The graph architectures are still relatively shallow, but the current `GraphNET` and `GraphMP` are materially richer than the earlier baseline descriptions because they use edge-aware message passing and stronger readouts.
5. Every graph model concatenates pooled graph information with the `12`-D global RDKit descriptor vector before the final regression head.
6. The dense neural loops save the best validation checkpoint, but they still run through all epochs.
7. Full-pipeline runtime is strongly affected by Gaussian process training.
8. The scripted graph tuning still excludes `GraphMP`; the best `GraphMP` result so far came from manual CUDA tuning on `node1`.

### 16. Presentation Q&A Cheat Sheet

Quick answers for common viva/presentation questions.

**Why regression first?**
Because the original target is continuous solubility, so regression preserves the actual scientific quantity.

**Why classification in ablation?**
Because it creates a simpler decision problem and lets us compare all families on the same binary target.

**Why use both classical and deep learning?**
Because classical models give interpretable baselines, while neural models capture nonlinear and structural information.

**Why is GPR slow?**
Because exact Gaussian process regression scales cubically with the number of training samples due to kernel matrix factorization, not because it has many parameters.

**Why graph models?**
Because molecules are naturally graphs, and graph networks can use atom-bond structure directly rather than only handcrafted tabular features.

**Why can a DNN be faster than GPR despite more parameters?**
Because parameter count is not the only factor. DNNs use minibatch gradient descent, while GPR must solve a large dense kernel system over all training samples.

### 17. Big Picture Takeaway

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

# AqSolDB Experiments Overview Guide

This document explains the notebook [aqsoldb_experiments_overview.ipynb](/home/priyank1/Palash/FOML_GOD/FoML_Project/notebooks/aqsoldb_experiments_overview.ipynb) section by section so you can understand both the preprocessing and the experiments without having to jump across many Python files.

## Goal

The notebook was created to consolidate the repository workflow into one place. It is designed to help you explain the project in a report, viva, presentation, or self-study session.

It covers:

- the cleaned AqSolDB dataset
- tabular feature generation
- graph construction
- train/test and validation splitting
- regression experiments from `train/`
- tuning experiments from `tuning/`
- binary classification ablation studies from `ablation/`
- where outputs are saved and how to compare them

## What the notebook is doing

The notebook does not invent a new pipeline. It reuses the same project functions that the scripts already use. That is important because it means:

- the notebook matches the repository logic
- the explanations stay faithful to the codebase
- you can read the notebook and trust that it reflects the actual experiments

In other words, the notebook is a readable wrapper around the project.

## Section-by-section explanation

### 1. Setup

This section makes sure the notebook can import the repository modules correctly.

Why this matters:

- Jupyter notebooks are often launched from different folders
- imports fail if the repository root is not on `sys.path`
- the setup cell checks whether you are in the project root or inside `notebooks/`

It also imports the main helpers used throughout the notebook:

- `load_dataset`
- `build_classical_feature_matrix`
- `build_graph_dataset`
- `make_binary_labels`
- `split_classical_data`
- `regression_metrics`
- `classification_metrics`

So after this cell, the notebook is ready to inspect preprocessing and run experiments.

### 2. Load and clean the dataset

This section shows the first step of the full pipeline.

It compares:

- the raw CSV loaded directly from `data/curated-solubility-dataset.csv`
- the cleaned dataset returned by `load_dataset()`

The cleaning logic comes from [data_utils.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/utils/data_utils.py). That function:

- reads the CSV
- verifies that `Name`, `SMILES`, and `Solubility` are present
- converts `Solubility` to numeric values
- drops rows with invalid target values
- parses each SMILES string with RDKit
- removes invalid molecules
- stores canonical SMILES in a new `CanonicalSMILES` column

The notebook then prints a compact summary:

- number of raw rows
- number of clean rows
- number of dropped rows
- mean, standard deviation, minimum, and maximum solubility

This section is useful because it makes the preprocessing visible before any model is trained.

### 3. Tabular preprocessing

This section explains the representation used by:

- linear regression
- Gaussian process regression
- scikit-learn MLP regressor
- PyTorch dense regressor
- the tabular ablation models

The function `build_classical_feature_matrix()` converts every valid molecule into a fixed-length numeric vector.

The notebook runs all three feature modes:

- `fingerprint`
- `descriptor`
- `combined`

#### Fingerprint mode

This uses Morgan fingerprints only.

Interpretation:

- captures local structural patterns in the molecule
- gives a high-dimensional bit vector
- useful for classical ML and DNN inputs

#### Descriptor mode

This uses the RDKit descriptor list defined in `DEFAULT_DESCRIPTOR_NAMES`.

These descriptors summarize global physicochemical properties such as:

- molecular weight
- logP
- polar surface area
- ring counts
- aromatic ring counts
- hydrogen donor counts

#### Combined mode

This concatenates:

- Morgan fingerprint
- RDKit descriptors

This is the default representation used in the main regression scripts. It is the richest tabular representation in the repository because it combines local structural information and global chemical properties.

The notebook also prints:

- how many rows are usable
- the number of features in each mode
- the descriptor names
- a sample of the combined feature names

That makes the feature engineering much easier to explain than reading it from scattered scripts.

### 4. Graph preprocessing

This section explains how the project turns molecules into graphs for graph neural networks.

The function `build_graph_dataset()` creates one PyTorch Geometric `Data` object per molecule.

Each graph contains:

- `x`: node features
- `edge_index`: connectivity
- `edge_attr`: edge weights based on bond type
- `y`: the target value

#### Node features

The notebook shows the three graph feature variants used in the repository:

- `atomic_number`
- `atomic_number_degree`
- `full`

Interpretation:

- `atomic_number` is the smallest feature set
- `atomic_number_degree` adds local connectivity information
- `full` adds aromaticity and is the default graph representation for the main regression scripts

#### Edge information

The preprocessing code also builds `edge_attr` from bond types:

- single bond -> `1.0`
- double bond -> `2.0`
- triple bond -> `3.0`
- fallback or aromatic-style case -> `1.5`

Important project note:

- these edge attributes are prepared in preprocessing
- the current graph models do not explicitly consume `edge_attr` in their forward pass

So the bond weights exist in the dataset objects, but the current architectures are effectively using only node features and adjacency structure.

The notebook summarizes graph statistics such as:

- number of graphs
- node feature dimension
- average and median atom count
- average and median directed edge count

It also prints one sample graph object so you can see the internal structure directly.

### 5. Shared split logic

This section explains how the data is split before training.

For the main tabular regression setup:

- `80%` of the data goes to training
- `20%` goes to testing

This is handled by `split_classical_data()`.

For dense neural networks and graph neural networks, there is one more split:

- the training portion is split again into fit and validation subsets

This second split is used for model selection during training, especially for tracking validation loss and keeping the best model state.

The notebook includes a small demo of the default tabular split so you can see:

- train rows
- test rows
- feature dimension

This helps connect preprocessing with the actual experiments.

### 6. Regression experiments

This is the main experiment section for the project.

It connects directly to the scripts in `train/`:

- [train_linear_regression.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/train/train_linear_regression.py)
- [train_gaussian_process.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/train/train_gaussian_process.py)
- [train_mlp_regressor.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/train/train_mlp_regressor.py)
- [train_dense_regressor.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/train/train_dense_regressor.py)
- [train_graph_cn.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/train/train_graph_cn.py)
- [train_graph_net.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/train/train_graph_net.py)
- [train_graph_sage.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/train/train_graph_sage.py)
- [train_graph_mp.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/train/train_graph_mp.py)

The notebook imports each `train_and_evaluate()` function and provides run flags:

- `RUN_FULL_REGRESSION`
- `RUN_GAUSSIAN_PROCESS`

This was done deliberately.

Why:

- the notebook should still be readable without forcing long runs
- Gaussian process regression can be expensive on this dataset
- neural experiments may also take noticeable time depending on hardware

So the section is ready to run all experiments, but you can control when to execute them.

#### What each regression model is doing

`linear_regression`

- simplest baseline
- assumes solubility is a linear function of the tabular features
- useful as a reference point

`gaussian_process`

- nonlinear probabilistic regression model
- uses an RBF kernel
- more expressive than linear regression
- can be much slower and heavier on memory

`mlp_regressor`

- scikit-learn multilayer perceptron
- uses standardized tabular features
- good midpoint between classical and deep learning approaches

`dense_regressor`

- PyTorch feed-forward neural network
- scales the tabular features with `StandardScaler`
- tracks training and validation loss
- saves a learned state dictionary plus training history

`graph_cn`

- graph convolution style model using message passing

`graph_net`

- attention-style message passing model

`graph_sage`

- neighborhood aggregation inspired by GraphSAGE

`graph_mp`

- custom message-passing model that concatenates source and target node features when building messages

All these models ultimately predict one scalar solubility value per molecule.

#### Output metrics

The regression scripts save and report:

- RMSE
- MAE
- R²

These are computed through [metrics.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/utils/metrics.py).

### 7. Hyperparameter tuning studies

This section groups the scripts in `tuning/` into one notebook area.

The notebook exposes a single flag:

- `RUN_TUNING`

If you enable it, the notebook runs:

- `tune_classical_models()`
- `tune_dnn()`
- `tune_graph_models()`

#### Classical tuning

The classical tuning script:

- uses linear regression as a baseline
- performs randomized search for Gaussian process parameters
- performs randomized search for MLP regressor parameters

The results are saved to:

- `outputs/tuning/classical_tuning_results.csv`

#### DNN tuning

The DNN tuning script compares a small list of dense network configurations:

- different hidden layer sizes
- different dropout values
- different learning rates

The results are saved to:

- `outputs/tuning/dnn_tuning_results.csv`

#### Graph tuning

The graph tuning script compares:

- graph architecture
- hidden channel width
- learning rate

The results are saved to:

- `outputs/tuning/graph_tuning_results.csv`

This section is helpful because the tuning experiments are conceptually separate from the base training scripts, but still part of the complete project story.

### 8. Ablation studies

This section explains the second major study in the repository.

Unlike the main pipeline, the ablation folder treats the problem as binary classification.

The threshold rule is:

- class `1` if `Solubility >= -3.0`
- class `0` otherwise

The notebook first computes the positive-class rate so you can see the class balance clearly.

Then it provides a run flag:

- `RUN_ABLATION`

If enabled, it executes:

- `run_feature_ablation()`
- `run_graph_ablation()`

#### Feature ablation

This compares tabular models across:

- fingerprint-only features
- descriptor-only features
- combined features

The evaluated models are:

- logistic regression
- Gaussian process classifier
- MLP classifier
- dense classifier

The results are saved to:

- `outputs/ablation/tabular_classification_ablation.csv`

#### Graph ablation

This compares graph models across node-feature variants:

- `atomic_number`
- `atomic_number_degree`
- `full`

The evaluated graph models are:

- GraphCN
- GraphNET
- GraphSAGE
- GraphMP

The results are saved to:

- `outputs/ablation/graph_classification_ablation.csv`

#### Classification metrics

The classification ablation scripts compute:

- accuracy
- precision
- recall
- F1
- ROC-AUC

This section is especially important in presentations because it shows that the repository is not only about model comparison, but also about studying how representation and task formulation change the results.

### 9. Read saved outputs in one place

This is the reporting section of the notebook.

The notebook defines a helper that walks through `outputs/` and collects every `metrics.json` file into one table.

Why this is useful:

- after you run multiple experiments, files get scattered under different subfolders
- this section gives you one compact comparison table
- it helps you discuss model performance without manually opening each output directory

If no outputs exist yet, the notebook tells you that clearly instead of failing.

### 10. Suggested workflow

The final notebook section gives a recommended order of use:

1. run setup and preprocessing
2. inspect tabular and graph representations
3. run regression experiments
4. run tuning
5. run ablation
6. collect outputs for comparison

This section is there to make the notebook easy to use for someone who did not write the codebase.

## How this notebook maps to the repository

This is the most important mapping:

- preprocessing logic comes from [data_utils.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/utils/data_utils.py)
- shared training loops come from [training_utils.py](/home/priyank1/Palash/FOML_GOD/FoML_Project/utils/training_utils.py)
- tabular baselines come from [ClassicalModels](/home/priyank1/Palash/FOML_GOD/FoML_Project/ClassicalModels)
- dense neural models come from [DNN](/home/priyank1/Palash/FOML_GOD/FoML_Project/DNN)
- graph models come from [GraphML](/home/priyank1/Palash/FOML_GOD/FoML_Project/GraphML)
- regression experiments come from [train](/home/priyank1/Palash/FOML_GOD/FoML_Project/train)
- tuning studies come from [tuning](/home/priyank1/Palash/FOML_GOD/FoML_Project/tuning)
- ablation studies come from [ablation](/home/priyank1/Palash/FOML_GOD/FoML_Project/ablation)

So if someone asks, "Where is this shown in code?", the notebook gives the readable explanation and the repository still gives the implementation details.

## Important practical note

At the time this notebook and guide were created, the repository did not contain an `outputs/` directory yet. That means:

- the notebook includes runnable experiment cells
- the final output-comparison section is ready
- but actual saved metrics will appear only after the experiments are executed

This is why some cells are controlled by flags instead of running everything automatically.

## Recommended way to present this project

If you want to explain the project clearly to someone else, the strongest order is:

1. start with the cleaned dataset
2. explain the two molecule representations: tabular and graph
3. explain the main regression pipeline
4. compare classical, dense, and graph models
5. mention tuning as refinement
6. end with the ablation study as a secondary classification analysis

That sequence matches both the logic of the codebase and the flow of the notebook.

## Deliverables created

The following files were added:

- [aqsoldb_experiments_overview.ipynb](/home/priyank1/Palash/FOML_GOD/FoML_Project/notebooks/aqsoldb_experiments_overview.ipynb)
- [aqsoldb_experiments_overview.md](/home/priyank1/Palash/FOML_GOD/FoML_Project/notebooks/aqsoldb_experiments_overview.md)

These two files should give you a much easier single-place view of the preprocessing and all experiment types in the project.

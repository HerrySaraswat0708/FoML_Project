# AqSolDB Presentation Summary

## 1. Project Goal

This project predicts **aqueous solubility** of small molecules from their **SMILES** strings using three model families:

- **Classical machine learning**
- **Tabular deep neural networks (DNN)**
- **Graph neural networks (GraphML / GNNs)**

The project is intentionally structured as a **course project**:

- the **main pipeline** is a **regression problem**
- the **ablation study** converts the target into a **binary classification problem**
- the full focus is on **hands-on learning**, **end-to-end pipeline design**, and **theoretical understanding of model behavior**

---

## 2. What We Are Predicting

The dataset is AqSolDB. The main target column is:

- `Solubility` (continuous)

This means the main task is:

- input: molecule
- output: a real number
- learning type: **supervised regression**

For the ablation study, we convert the continuous target into a binary label:

- positive class: `Solubility >= -3.0`
- negative class: `Solubility < -3.0`

So ablation is:

- input: molecule
- output: class label
- learning type: **binary classification**

---

## 3. Current Project Structure

- `ClassicalModels/`
  - regression and classification builders for linear/logistic regression, Gaussian process, and MLP
- `DNN/`
  - dense neural network for regression and classification
- `GraphML/`
  - graph neural network architectures: `GraphCN`, `GraphNET`, `GraphSAGE`, `GraphMP`
- `train/`
  - end-to-end **regression** training scripts
- `tuning/`
  - hyperparameter search scripts
- `ablation/`
  - **binary classification** comparison scripts
- `utils/`
  - data loading, feature generation, metrics, and training loops
- `outputs/`
  - metrics, predictions, weights, histories, study tables

---

## 4. End-to-End Pipeline

The project follows a standard ML pipeline:

1. **Load and clean data**
2. **Convert molecules into model-ready representations**
3. **Split data into train / validation / test**
4. **Train the model**
5. **Evaluate on held-out data**
6. **Save metrics, predictions, and model artifacts**

This is implemented through:

- `utils/data_utils.py`
- `utils/training_utils.py`
- model-specific scripts in `train/`

---

## 5. Data Representation

### 5.1 Tabular representation

For classical ML and tabular DNNs, each molecule is converted into a fixed-length feature vector:

- **Morgan fingerprint** of length `1024`
- **12 RDKit descriptors**

So the default feature vector length is:

- `1024 + 12 = 1036`

On the cleaned dataset:

- valid molecules: **9980**
- tabular feature matrix shape: **(9980, 1036)**

This representation is useful because classical models and dense neural networks require **fixed-size numeric vectors**.

### 5.2 Graph representation

For graph models, each molecule becomes a graph:

- nodes = atoms
- edges = bonds
- whole graph = one molecule

Default node features:

- atomic number
- degree
- aromaticity flag

So graph learning here is **graph-level prediction**, not node classification.

Dataset graph statistics:

- number of graphs: **9980**
- average atoms per molecule: **17.38**
- median atoms per molecule: **15**
- average directed edges per molecule: **35.27**
- median directed edges per molecule: **30**

---

## 6. Why Regression Is the Main Pipeline

The original target is continuous solubility, so regression is the most natural setup.

### Regression objective

Learn a function:

`f(x) -> y`

where:

- `x` = molecule representation
- `y` = real-valued solubility

Typical regression loss:

- **Mean Squared Error (MSE)**

Formula:

`MSE = (1/n) * sum((y_i - yhat_i)^2)`

Regression is used in:

- linear regression
- Gaussian process regression
- MLP regressor
- dense regressor
- graph regressors

---

## 7. Why Classification Is Used in Ablation

Ablation is meant to answer:  
**What changes when we simplify the target into a decision problem?**

We define:

- class `1` if `Solubility >= -3.0`
- class `0` otherwise

On the cleaned dataset, positive class rate is:

- **0.5696** (about **56.96%**)

This makes classification a useful secondary study because:

- it is easier to interpret in practical terms
- it lets us compare all model families on the same discrete decision task
- it shows how performance changes when we throw away fine-grained target information

---

## 8. Model Families and Theory

### 8.1 Linear Regression

Idea:

- assume the prediction is a linear combination of features

Formula:

`yhat = w^T x + b`

Why it is useful:

- simplest baseline
- highly interpretable
- fast to train

Limitation:

- cannot model complex nonlinear molecular effects

### 8.2 Logistic Regression

Used in classification ablation.

Formula:

`p(y=1|x) = sigmoid(w^T x + b)`

Why it is useful:

- strong linear classification baseline
- fast and easy to interpret

### 8.3 Gaussian Process Regression (GPR)

Idea:

- instead of directly learning one fixed parametric equation, assume the target values are sampled from a **Gaussian process**
- similarity between molecules is measured by a **kernel**

In this project:

- kernel = **RBF**

RBF kernel:

`k(x, x') = exp(-||x - x'||^2 / (2 * l^2))`

where `l` is the length scale.

Why it is attractive:

- elegant probabilistic model
- strong small-data baseline
- can model smooth nonlinear structure

Why it becomes very slow:

- GPR needs an `n x n` kernel matrix over training points
- training complexity is roughly **O(n^3)**
- memory is roughly **O(n^2)**

With this project’s default split:

- total clean rows = `9980`
- training rows ≈ `7984`
- kernel matrix size = `7984 x 7984`
- entries = **63,744,256**
- float64 storage for that matrix alone is about **0.51 GB**

This is the main reason GPR is slow.  
It is **not** slow because it has many trainable parameters.  
It is slow because exact GP inference requires expensive matrix factorization.

### 8.4 MLP Regressor / MLP Classifier

Idea:

- a multilayer perceptron stacks linear layers and nonlinear activations

General layer formula:

`h_(l+1) = sigma(W_l h_l + b_l)`

Why it is stronger than linear regression:

- learns nonlinear combinations of features
- can model interactions between descriptors and fingerprints

Why it is still faster than GPR:

- its cost scales mainly with parameter count, batch size, and epochs
- it does **not** build an `n x n` kernel matrix

### 8.5 DenseRegressor / DenseClassifier

This is the PyTorch DNN version of the MLP idea.

Architecture:

- input `1036`
- hidden layers `256 -> 128 -> 64`
- output `1`

Why it exists even though there is already scikit MLP:

- gives hands-on exposure to custom neural network training
- makes backpropagation, minibatches, validation splits, and parameter counting clearer

### 8.6 Graph Neural Networks

Graph models operate directly on molecular structure.

General message-passing idea:

`h_v^(k+1) = Update(h_v^(k), Aggregate({Message(h_v^(k), h_u^(k)) for u in N(v)}))`

Then the node features are pooled to produce one graph-level output:

- here we use `global_mean_pool`

This is correct because:

- a molecule is one graph
- we want one prediction per molecule

So these are **graph-level regression** models in the main pipeline and **graph-level classification** models in ablation.

#### GraphCN

- graph convolution style
- normalizes neighbor aggregation using degree information

#### GraphNET

- attention-style message passing
- learns weights for different neighbors

#### GraphSAGE

- aggregates neighbor messages, then concatenates with self-information

#### GraphMP

- custom message passing with explicit message and update linear layers

Why graph models are important:

- they preserve connectivity information
- they use molecular structure more directly than fixed tabular descriptors

---

## 9. Parameter Counts

Parameter counts help explain training speed for parametric models, but they do **not** explain everything.

### 9.1 Tabular models

Default tabular input dimension:

- `1036`

#### Linear Regression

Formula:

- weights: `1036`
- bias: `1`

Total:

- **1037 trainable parameters**

#### Logistic Regression

Binary classifier with the same input dimension:

- **1037 trainable parameters**

#### Gaussian Process Regression

Exact GPR does **not** behave like a standard neural network parameter-count story.

Practical view:

- optimized kernel hyperparameters: essentially **1** (`length_scale`) in current setup
- stored dual coefficients depend on training size

Important point:

- GPR runtime is dominated by **kernel matrix inversion/factorization**, not parameter count

#### MLP Regressor / MLP Classifier

Default hidden sizes:

- `1036 -> 256 -> 128 -> 64 -> 1`

Parameter calculation:

- `1036*256 + 256 = 265472`
- `256*128 + 128 = 32896`
- `128*64 + 64 = 8256`
- `64*1 + 1 = 65`

Total:

- **306,689 trainable parameters**

#### DenseRegressor / DenseClassifier

Same default architecture as above:

- **306,689 trainable parameters**

### 9.2 Graph models

Default graph input dimension:

- node feature dimension = `3`
- hidden channels = `64`

#### GraphCN

Total:

- **321 trainable parameters**

#### GraphNET

Total:

- **385 trainable parameters**

#### GraphSAGE

Total:

- **513 trainable parameters**

#### GraphMP

Total:

- **4,673 trainable parameters**

### 9.3 Interpretation of parameter counts

Some important lessons:

1. **Few parameters does not always mean fast**
   - GPR has very few hyperparameters but is still very slow because of matrix operations over all training samples

2. **More parameters does not always mean slow**
   - DNNs can have hundreds of thousands of parameters but still train efficiently because they use gradient-based optimization and minibatches

3. **Graph models here are relatively small**
   - despite using neural networks, the graph models have small parameter counts because the hidden dimension is only `64` and the architecture is shallow

---

## 10. Why Some Models Are Fast and Others Are Slow

### Linear / Logistic Regression

Usually fast because:

- convex optimization
- low parameter count
- simple forward pass

### GPR

Slow because:

- exact kernel methods scale badly with dataset size
- training needs large matrix operations
- kernel hyperparameter optimization repeats expensive computation

### MLP / Dense DNN

Moderate speed because:

- many parameters
- multiple epochs
- backpropagation

Still manageable because:

- minibatch training avoids full `n x n` matrix costs

### Graph Models

Moderate to slow depending on:

- number of graphs
- average number of atoms/bonds
- number of epochs
- batch size

Graph models do more structural computation than tabular models, but their parameter counts here are still small.

---

## 11. Training Splits

Default main pipeline split:

- test size = `0.2`

So approximately:

- train+validation = `7984`
- test = `1996`

For DNN and graph models, an additional validation split is taken from the training portion:

- fit ≈ `6786`
- validation ≈ `1198`
- test = `1996`

This helps:

- tune training behavior
- reduce overfitting
- keep test data untouched until final evaluation

---

## 12. Metrics

### Regression metrics

- **RMSE**
  - penalizes large errors strongly
- **MAE**
  - average absolute difference between prediction and target
- **R²**
  - proportion of variance explained

### Classification metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 score**
- **ROC-AUC**

Why multiple metrics matter:

- one metric rarely tells the full story
- for classification, F1 and ROC-AUC are often more informative than accuracy alone

---

## 13. What the Project Teaches

This project is valuable because it shows three different views of the same molecular prediction problem:

1. **Feature-engineered classical ML**
2. **Feature-based deep learning**
3. **Structure-aware graph learning**

It also demonstrates a full ML workflow:

- data cleaning
- representation design
- train/validation/test splits
- optimization
- evaluation
- ablation
- tuning
- result logging

---

## 14. Key Talking Points for Presentation

### If asked: “Why regression first?”

Because the original target is continuous solubility, so regression preserves the actual scientific quantity.

### If asked: “Why classification in ablation?”

Because it creates a simpler decision problem and lets us compare all families on the same binary target.

### If asked: “Why use both classical and deep learning?”

Because classical models give interpretable baselines, while neural models capture nonlinear and structural information.

### If asked: “Why is GPR slow?”

Because exact Gaussian process regression scales cubically with the number of training samples due to kernel matrix factorization, not because it has many parameters.

### If asked: “Why graph models?”

Because molecules are naturally graphs, and graph networks can use atom-bond structure directly rather than only handcrafted tabular features.

### If asked: “Why can a DNN be faster than GPR despite more parameters?”

Because parameter count is not the only factor. DNNs use minibatch gradient descent, while GPR must solve a large dense kernel system over all training samples.

---

## 15. Final Takeaway

The core lesson of this project is:

- **representation matters**
- **model family matters**
- **problem formulation matters**
- **computational complexity matters**

The same molecule can be viewed as:

- a vector of descriptors
- a fingerprint
- a graph

And each view changes:

- what the model can learn
- how expensive training becomes
- how interpretable the result is

That is the main learning trajectory of the project.

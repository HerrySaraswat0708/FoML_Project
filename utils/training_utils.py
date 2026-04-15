from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import build_prediction_frame, regression_metrics
from .project_paths import model_output_dir


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_sklearn_run(
    family: str,
    model_name: str,
    model,
    test_frame: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    extra_metadata: dict[str, object] | None = None,
):
    import joblib

    output_dir = model_output_dir(family, model_name)
    predictions = build_prediction_frame(test_frame, y_test, y_pred)
    metrics = regression_metrics(y_test, y_pred)
    if extra_metadata:
        metrics.update(extra_metadata)

    joblib.dump(model, output_dir / "model.joblib")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_json(output_dir / "metrics.json", metrics)
    return output_dir, metrics


def save_torch_run(
    family: str,
    model_name: str,
    state_dict: dict[str, object],
    history: list[dict[str, float]],
    test_frame: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    extra_metadata: dict[str, object] | None = None,
):
    import torch

    output_dir = model_output_dir(family, model_name)
    predictions = build_prediction_frame(test_frame, y_test, y_pred)
    metrics = regression_metrics(y_test, y_pred)
    if extra_metadata:
        metrics.update(extra_metadata)

    torch.save(state_dict, output_dir / "model.pt")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    save_json(output_dir / "metrics.json", metrics)
    return output_dir, metrics


def train_torch_regressor(
    model,
    train_features,
    train_targets,
    val_features,
    val_targets,
    epochs: int = 150,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32).view(-1, 1),
    )
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32).view(-1, 1).to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_features.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_features_tensor)
            val_loss = float(loss_fn(val_predictions, val_targets_tensor).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

    model.load_state_dict(best_state)
    return model.cpu(), history


def predict_torch_regressor(model, features: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(features, dtype=torch.float32))
    return predictions.view(-1).cpu().numpy()


def train_graph_regressor(
    model,
    train_dataset,
    val_dataset,
    epochs: int = 120,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
):
    import torch
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(predictions, batch.y.view(-1, 1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch.num_graphs
            total_train_graphs += batch.num_graphs

        train_loss = total_train_loss / total_train_graphs

        model.eval()
        total_val_loss = 0.0
        total_val_graphs = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                predictions = model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(predictions, batch.y.view(-1, 1))
                total_val_loss += loss.item() * batch.num_graphs
                total_val_graphs += batch.num_graphs

        val_loss = total_val_loss / total_val_graphs

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

    model.load_state_dict(best_state)
    return model.cpu(), history


def predict_graph_regressor(model, dataset, batch_size: int = 32) -> np.ndarray:
    import torch
    from torch_geometric.loader import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch.x, batch.edge_index, batch.batch)
            predictions.extend(output.view(-1).cpu().numpy().tolist())
    return np.asarray(predictions, dtype=np.float32)


def train_torch_binary_classifier(
    model,
    train_features,
    train_targets,
    val_features,
    val_targets,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32).view(-1, 1),
    )
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32).view(-1, 1).to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = loss_fn(logits, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_features.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_features_tensor)
            val_loss = float(loss_fn(val_logits, val_targets_tensor).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

    model.load_state_dict(best_state)
    return model.cpu(), history


def predict_torch_binary_classifier(model, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).view(-1).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    return predictions, probabilities


def train_graph_binary_classifier(
    model,
    train_dataset,
    val_dataset,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
):
    import torch
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits, batch.y.view(-1, 1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch.num_graphs
            total_train_graphs += batch.num_graphs

        train_loss = total_train_loss / total_train_graphs

        model.eval()
        total_val_loss = 0.0
        total_val_graphs = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(logits, batch.y.view(-1, 1))
                total_val_loss += loss.item() * batch.num_graphs
                total_val_graphs += batch.num_graphs

        val_loss = total_val_loss / total_val_graphs

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

    model.load_state_dict(best_state)
    return model.cpu(), history


def predict_graph_binary_classifier(model, dataset, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch_geometric.loader import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.x, batch.edge_index, batch.batch)
            probabilities.extend(torch.sigmoid(logits).view(-1).cpu().numpy().tolist())

    scores = np.asarray(probabilities, dtype=np.float32)
    predictions = (scores >= 0.5).astype(np.int64)
    return predictions, scores

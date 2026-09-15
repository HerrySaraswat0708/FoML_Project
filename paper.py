#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_aqsol_gnns.py

Reproducible AqSolDB graph-regression pipeline using only:
- GraphCN
- GraphNET
- GraphSAGE
- GraphMP

What it does
------------
1. Loads AqSolDB from CSV (or auto-downloads a common public copy if missing)
2. Parses SMILES with RDKit
3. Builds PyTorch Geometric graph objects
4. Creates Bemis-Murcko scaffold split (8:1:1)
5. Trains selected graph model with 4 seeds
6. Reports MAE / RMSE / R2
7. Saves per-seed metrics and predictions

Example
-------
python train_aqsol_gnns.py --model all --csv_path AqSolDB.csv --epochs 200 --batch_size 64

Requirements
------------
pip install torch torchvision torchaudio
pip install torch_geometric
pip install rdkit-pypi pandas numpy scikit-learn requests tqdm

Notes
-----
- This is designed to be self-contained and easy to modify.
- It is faithful to the *experiment style* of the benchmark paper:
  graph-level regression on AqSolDB with scaffold split and MAE loss.
- It does NOT guarantee the exact numbers from the paper because the official
  benchmark code/configs are different and the requested models differ from the paper's exact list.
"""

from __future__ import annotations

import os
import json
import math
import time
import random
import argparse
import warnings
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, softmax, degree

warnings.filterwarnings("ignore")


# =========================
# Reproducibility utilities
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# AqSolDB loading
# =========================

DEFAULT_AQSOL_URLS = [
    # Try a few commonly used mirrors / raw files.
    "https://raw.githubusercontent.com/tsorkun/AqSolDB/master/data/AqSolDB.csv",
    "https://raw.githubusercontent.com/tsorkun/AqSolDB/master/AqSolDB.csv",
]

POSSIBLE_SMILES_COLS = [
    "SMILES", "smiles", "CanonicalSMILES", "canonical_smiles"
]

POSSIBLE_TARGET_COLS = [
    "Solubility", "solubility",
    "Solubility_logS", "logS", "LogS", "Measured LogS"
]


def maybe_download_csv(csv_path: str) -> str:
    if os.path.exists(csv_path):
        return csv_path

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    for url in DEFAULT_AQSOL_URLS:
        try:
            print(f"[INFO] Trying download: {url}")
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 1000:
                with open(csv_path, "wb") as f:
                    f.write(r.content)
                print(f"[INFO] Downloaded AqSolDB to: {csv_path}")
                return csv_path
        except Exception as e:
            print(f"[WARN] Download failed from {url}: {e}")

    raise FileNotFoundError(
        f"Could not find or download AqSolDB CSV.\n"
        f"Please place the dataset locally and pass --csv_path.\n"
        f"Tried: {DEFAULT_AQSOL_URLS}"
    )


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    smiles_col = None
    target_col = None

    for c in POSSIBLE_SMILES_COLS:
        if c in df.columns:
            smiles_col = c
            break

    for c in POSSIBLE_TARGET_COLS:
        if c in df.columns:
            target_col = c
            break

    if smiles_col is None:
        raise ValueError(
            f"Could not find SMILES column. Available columns: {list(df.columns)}"
        )
    if target_col is None:
        raise ValueError(
            f"Could not find target column. Available columns: {list(df.columns)}"
        )

    return smiles_col, target_col


# =========================
# Chemistry featurization
# =========================

BOND_TYPE_TO_INT = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
UNKNOWN_BOND_TYPE = 4


def safe_scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def atom_symbol_vocab(mols: List[Chem.Mol]) -> Dict[str, int]:
    symbols = set()
    for mol in mols:
        for atom in mol.GetAtoms():
            symbols.add(atom.GetSymbol())
    symbols = sorted(symbols)
    return {sym: i for i, sym in enumerate(symbols)}


def atom_features(atom: Chem.Atom, symbol_to_idx: Dict[str, int]) -> List[float]:
    """
    A moderately informative numeric feature vector.
    Keeps the code simple and model-agnostic.
    """
    symbol_idx = symbol_to_idx.get(atom.GetSymbol(), 0)
    feat = [
        float(symbol_idx),
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(int(atom.GetIsAromatic())),
        float(atom.GetTotalNumHs()),
        float(int(atom.IsInRing())),
        float(atom.GetMass() / 200.0),
    ]
    return feat


def mol_to_graph(
    mol: Chem.Mol,
    y: float,
    symbol_to_idx: Dict[str, int],
    smiles: str,
) -> Optional[Data]:
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:
        return None
    if mol.GetNumBonds() == 0:
        return None

    x_list = [atom_features(atom, symbol_to_idx) for atom in mol.GetAtoms()]
    x = torch.tensor(x_list, dtype=torch.float)

    edge_index_list = []
    edge_attr_list = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BOND_TYPE_TO_INT.get(bond.GetBondType(), UNKNOWN_BOND_TYPE)

        edge_index_list.append([i, j])
        edge_index_list.append([j, i])

        edge_attr_list.append([float(bond_type)])
        edge_attr_list.append([float(bond_type)])

    if len(edge_index_list) == 0:
        return None

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(y)], dtype=torch.float),
        smiles=smiles,
    )
    return data


def load_aqsol_graphs(csv_path: str) -> List[Data]:
    csv_path = maybe_download_csv(csv_path)
    df = pd.read_csv(csv_path)

    smiles_col, target_col = detect_columns(df)
    print(f"[INFO] Using SMILES column: {smiles_col}")
    print(f"[INFO] Using target column: {target_col}")

    df = df[[smiles_col, target_col]].dropna().copy()
    df = df.rename(columns={smiles_col: "smiles", target_col: "target"})
    df["smiles"] = df["smiles"].astype(str)

    mols = []
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing molecules"):
        smi = row["smiles"]
        y = row["target"]
        try:
            y = float(y)
        except Exception:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if mol.GetNumAtoms() == 0 or mol.GetNumBonds() == 0:
            continue
        mols.append(mol)
        rows.append((smi, y, mol))

    symbol_to_idx = atom_symbol_vocab([m for _, _, m in rows])
    print(f"[INFO] Unique atom symbols: {len(symbol_to_idx)}")

    dataset = []
    for smi, y, mol in tqdm(rows, desc="Building graphs"):
        g = mol_to_graph(mol, y, symbol_to_idx, smi)
        if g is not None:
            dataset.append(g)

    print(f"[INFO] Final graph count: {len(dataset)}")
    return dataset


# =========================
# Scaffold split
# =========================

def scaffold_split(
    dataset: List[Data],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-8

    scaffold_to_indices = defaultdict(list)
    for i, data in enumerate(dataset):
        scaffold = safe_scaffold(data.smiles)
        scaffold_to_indices[scaffold].append(i)

    scaffold_sets = sorted(
        scaffold_to_indices.values(),
        key=lambda inds: (len(inds), inds[0]),
        reverse=True,
    )

    n_total = len(dataset)
    train_cutoff = int(frac_train * n_total)
    valid_cutoff = int((frac_train + frac_valid) * n_total)

    train_idx, valid_idx, test_idx = [], [], []

    for inds in scaffold_sets:
        if len(train_idx) + len(inds) <= train_cutoff:
            train_idx.extend(inds)
        elif len(train_idx) + len(valid_idx) + len(inds) <= valid_cutoff:
            valid_idx.extend(inds)
        else:
            test_idx.extend(inds)

    return train_idx, valid_idx, test_idx


# =========================
# Models
# =========================

class GraphCNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GraphCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphCNLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(GraphCNLayer(hidden_channels, hidden_channels))
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.head(g).view(-1)


class GraphNETLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add", node_dim=0)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Linear(2 * out_channels, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x) + self.bias

    def message(self, x_i, x_j, index):
        alpha = self.att(torch.cat([x_i, x_j], dim=-1))
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)
        return alpha * x_j


class GraphNET(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphNETLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(GraphNETLayer(hidden_channels, hidden_channels))
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.head(g).view(-1)


class GraphSAGELayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        neigh = self.propagate(edge_index, x=x)
        out = self.lin_self(x) + self.lin_neigh(neigh)
        return out

    def message(self, x_j):
        return x_j


class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(GraphSAGELayer(hidden_channels, hidden_channels))
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.head(g).view(-1)


class GraphMPLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.upd = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x, edge_index):
        m = self.propagate(edge_index, x=x)
        out = self.upd(torch.cat([x, m], dim=-1))
        return out

    def message(self, x_i, x_j):
        return self.msg_mlp(torch.cat([x_i, x_j], dim=-1))


class GraphMP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            [GraphMPLayer(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.head(g).view(-1)


def build_model(model_name: str, in_channels: int, hidden_channels: int, num_layers: int, dropout: float) -> nn.Module:
    name = model_name.lower()
    if name == "graphcn":
        return GraphCN(in_channels, hidden_channels, num_layers, dropout)
    if name == "graphnet":
        return GraphNET(in_channels, hidden_channels, num_layers, dropout)
    if name == "graphsage":
        return GraphSAGE(in_channels, hidden_channels, num_layers, dropout)
    if name == "graphmp":
        return GraphMP(in_channels, hidden_channels, num_layers, dropout)
    raise ValueError(f"Unknown model: {model_name}")


# =========================
# Training / evaluation
# =========================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, preds = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        ys.append(batch.y.view(-1).cpu().numpy())
        preds.append(out.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        y = batch.y.view(-1)
        loss = F.l1_loss(pred, y)   # matches paper's graph-regression MAE training style
        loss.backward()
        optimizer.step()

        n = batch.num_graphs
        total_loss += loss.item() * n
        total_graphs += n

    return total_loss / max(total_graphs, 1)


def run_single_seed(
    dataset: List[Data],
    train_idx: List[int],
    valid_idx: List[int],
    test_idx: List[int],
    args,
    seed: int,
    model_name: str,
) -> Dict[str, float]:
    set_seed(seed)
    device = get_device()

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(Subset(dataset, valid_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    in_channels = dataset[0].x.size(1)
    model = build_model(
        model_name=model_name,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, valid_loader, device)
        val_mae = val_metrics["mae"]
        scheduler.step(val_mae)

        current_lr = optimizer.param_groups[0]["lr"]

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"[{model_name}][seed={seed}] "
                f"epoch={epoch:03d} train_L1={train_loss:.4f} "
                f"val_MAE={val_mae:.4f} lr={current_lr:.2e}"
            )

        if current_lr <= args.min_lr + 1e-12:
            print(f"[{model_name}][seed={seed}] stopping: lr reached min_lr")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best checkpoint captured.")

    model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, device)
    valid_metrics = evaluate(model, valid_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - start

    result = {
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_mae": train_metrics["mae"],
        "val_mae": valid_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "elapsed_sec": elapsed,
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "y_true": test_metrics["y_true"],
        "y_pred": test_metrics["y_pred"],
    }
    return result


# =========================
# Saving helpers
# =========================

def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_predictions(path: str, y_true: List[float], y_pred: List[float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.to_csv(path, index=False)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="AqSolDB.csv")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "graphcn", "graphnet", "graphsage", "graphmp"])
    parser.add_argument("--output_dir", type=str, default="outputs_aqsol_paper_style")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    print("[INFO] Loading dataset...")
    dataset = load_aqsol_graphs(args.csv_path)

    print("[INFO] Building scaffold split...")
    train_idx, valid_idx, test_idx = scaffold_split(dataset, 0.8, 0.1, 0.1)
    print(f"[INFO] Split sizes: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}")

    models = ["graphcn", "graphnet", "graphsage", "graphmp"] if args.model == "all" else [args.model]

    all_rows = []

    for model_name in models:
        print(f"\n========== MODEL: {model_name} ==========")
        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        seed_results = []
        for seed in args.seeds:
            result = run_single_seed(
                dataset=dataset,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                args=args,
                seed=seed,
                model_name=model_name,
            )
            seed_results.append(result)

            save_json(os.path.join(model_dir, f"metrics_seed_{seed}.json"), result)
            save_predictions(
                os.path.join(model_dir, f"predictions_seed_{seed}.csv"),
                result["y_true"],
                result["y_pred"],
            )

            row = {k: v for k, v in result.items() if k not in ("y_true", "y_pred")}
            all_rows.append(row)

        test_maes = [r["test_mae"] for r in seed_results]
        test_rmses = [r["test_rmse"] for r in seed_results]
        test_r2s = [r["test_r2"] for r in seed_results]
        params = seed_results[0]["n_params"]

        summary = {
            "model": model_name,
            "n_params": params,
            "split": {
                "train": len(train_idx),
                "valid": len(valid_idx),
                "test": len(test_idx),
            },
            "test_mae_mean": float(np.mean(test_maes)),
            "test_mae_std": float(np.std(test_maes, ddof=1)) if len(test_maes) > 1 else 0.0,
            "test_rmse_mean": float(np.mean(test_rmses)),
            "test_rmse_std": float(np.std(test_rmses, ddof=1)) if len(test_rmses) > 1 else 0.0,
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s, ddof=1)) if len(test_r2s) > 1 else 0.0,
            "seeds": args.seeds,
        }

        save_json(os.path.join(model_dir, "summary.json"), summary)

        print(
            f"[SUMMARY][{model_name}] "
            f"MAE={summary['test_mae_mean']:.4f}±{summary['test_mae_std']:.4f} | "
            f"RMSE={summary['test_rmse_mean']:.4f}±{summary['test_rmse_std']:.4f} | "
            f"R2={summary['test_r2_mean']:.4f}±{summary['test_r2_std']:.4f} | "
            f"Params={params}"
        )

    results_df = pd.DataFrame(all_rows)
    results_path = os.path.join(args.output_dir, "all_seed_results.csv")
    results_df.to_csv(results_path, index=False)

    leaderboard = (
        results_df.groupby("model", as_index=False)
        .agg(
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            n_params=("n_params", "first"),
        )
        .sort_values("test_mae_mean", ascending=True)
    )
    leaderboard_path = os.path.join(args.output_dir, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)

    print("\n========== FINAL LEADERBOARD ==========")
    print(leaderboard.to_string(index=False))
    print(f"\n[INFO] Saved all results to: {args.output_dir}")


if __name__ == "__main__":
    main()
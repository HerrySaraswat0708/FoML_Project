import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .project_paths import DATASET_PATH, FEATURE_MATRIX_PATH, GRAPH_DATASET_PATH, PCA3D_DATASET_PATH, TARGET_VECTOR_PATH

DEFAULT_DESCRIPTOR_NAMES = (
    "NumHDonors",
    "TPSA",
    "NumRotatableBonds",
    "BertzCT",
    "RingCount",
    "NumAromaticRings",
    "NumValenceElectrons",
    "LabuteASA",
    "HeavyAtomCount",
    "MolWt",
    "MolMR",
    "MolLogP",
)

DESCRIPTOR_FUNCTIONS = {  # type: Dict[str, object]
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "MolMR": Descriptors.MolMR,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumSaturatedRings": Descriptors.NumSaturatedRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "RingCount": Descriptors.RingCount,
    "TPSA": Descriptors.TPSA,
    "LabuteASA": Descriptors.LabuteASA,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
}


def configure_runtime() -> None:
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    try:
        from rdkit import RDLogger
    except ImportError:
        return

    RDLogger.DisableLog("rdApp.warning")


def load_dataset(dataset_path=DATASET_PATH) -> pd.DataFrame:
    configure_runtime()

    frame = pd.read_csv(dataset_path)
    required_columns = {"Name", "SMILES", "Solubility"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame["Solubility"] = pd.to_numeric(frame["Solubility"], errors="coerce")
    frame = frame.dropna(subset=["Solubility"]).reset_index(drop=True)

    valid_rows = []  # type: List[int]
    canonical_smiles = []  # type: List[str]
    for index, smiles in frame["SMILES"].astype(str).items():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        valid_rows.append(index)
        canonical_smiles.append(Chem.MolToSmiles(molecule))

    cleaned = frame.loc[valid_rows].copy().reset_index(drop=True)
    cleaned["CanonicalSMILES"] = canonical_smiles
    return cleaned


def descriptor_vector(
    molecule: Chem.Mol,
    descriptor_names: Sequence[str] = DEFAULT_DESCRIPTOR_NAMES,
) -> np.ndarray:
    return np.asarray(
        [float(DESCRIPTOR_FUNCTIONS[name](molecule)) for name in descriptor_names],
        dtype=np.float32,
    )


def fingerprint_vector(
    molecule: Chem.Mol,
    fingerprint_radius: int = 2,
    fingerprint_size: int = 1024,
) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=fingerprint_radius,
        fpSize=fingerprint_size,
    )
    bits = np.zeros((fingerprint_size,), dtype=np.int8)
    ConvertToNumpyArray(generator.GetFingerprint(molecule), bits)
    return bits.astype(np.float32)


def build_classical_feature_matrix(
    frame: pd.DataFrame,
    descriptor_names: Sequence[str] = DEFAULT_DESCRIPTOR_NAMES,
    fingerprint_radius: int = 2,
    fingerprint_size: int = 1024,
    feature_mode: str = "combined",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    rows = []  # type: List[np.ndarray]
    targets = []  # type: List[float]
    valid_indices = []  # type: List[int]
    feature_names = []  # type: List[str]

    for index, row in frame.iterrows():
        molecule = Chem.MolFromSmiles(str(row["SMILES"]))
        if molecule is None:
            continue

        fingerprints = fingerprint_vector(
            molecule=molecule,
            fingerprint_radius=fingerprint_radius,
            fingerprint_size=fingerprint_size,
        )
        descriptors = descriptor_vector(molecule=molecule, descriptor_names=descriptor_names)

        if feature_mode == "fingerprint":
            feature_vector = fingerprints
            feature_names = [f"fp_{idx:04d}" for idx in range(fingerprint_size)]
        elif feature_mode == "descriptor":
            feature_vector = descriptors
            feature_names = list(descriptor_names)
        elif feature_mode == "combined":
            feature_vector = np.concatenate([fingerprints, descriptors])
            feature_names = [f"fp_{idx:04d}" for idx in range(fingerprint_size)] + list(descriptor_names)
        else:
            raise ValueError("feature_mode must be one of: fingerprint, descriptor, combined")

        rows.append(feature_vector)
        targets.append(float(row["Solubility"]))
        valid_indices.append(index)

    feature_frame = frame.loc[valid_indices].reset_index(drop=True)
    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return X, y, feature_frame, feature_names


def save_classical_arrays(
    X: np.ndarray,
    y: np.ndarray,
    feature_path=FEATURE_MATRIX_PATH,
    target_path=TARGET_VECTOR_PATH,
) -> None:
    np.save(feature_path, X)
    np.save(target_path, y)


def fit_pca_projection(
    train_features: np.ndarray,
    *other_feature_sets: np.ndarray,
    n_components: int = 3,
) -> Tuple[np.ndarray, ...]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    pca = PCA(n_components=n_components, random_state=42)
    train_projected = pca.fit_transform(train_scaled).astype(np.float32)

    transformed_sets = [train_projected]  # type: List[np.ndarray]
    for feature_set in other_feature_sets:
        projected = pca.transform(scaler.transform(feature_set)).astype(np.float32)
        transformed_sets.append(projected)

    feature_names = [f"pca_component_{index + 1}" for index in range(n_components)]
    transformed_sets.append(feature_names)
    transformed_sets.append(scaler)
    transformed_sets.append(pca)
    return tuple(transformed_sets)


def build_pca3d_dataset(
    frame: pd.DataFrame,
    source_feature_mode: str = "combined",
    n_components: int = 3,
) -> pd.DataFrame:
    X, y, clean_frame, _ = build_classical_feature_matrix(frame, feature_mode=source_feature_mode)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(X_scaled)

    report = clean_frame.copy()
    for index in range(n_components):
        report[f"pca_component_{index + 1}"] = projected[:, index]
    report["source_feature_mode"] = source_feature_mode
    report["pca_total_variance_explained"] = float(np.sum(pca.explained_variance_ratio_))
    return report


def save_pca3d_dataset(
    frame: pd.DataFrame,
    path=PCA3D_DATASET_PATH,
    source_feature_mode: str = "combined",
    n_components: int = 3,
) -> pd.DataFrame:
    dataset = build_pca3d_dataset(
        frame=frame,
        source_feature_mode=source_feature_mode,
        n_components=n_components,
    )
    dataset.to_csv(path, index=False)
    return dataset


def atom_feature_vector(atom: Chem.Atom, feature_variant: str = "full") -> List[float]:
    if feature_variant == "atomic_number":
        return [float(atom.GetAtomicNum())]
    if feature_variant == "atomic_number_degree":
        return [float(atom.GetAtomicNum()), float(atom.GetDegree())]
    if feature_variant == "full":
        hybridization = atom.GetHybridization()
        return [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            float(atom.GetImplicitValence()),
            float(atom.GetTotalValence()),
            float(int(atom.GetIsAromatic())),
            float(int(atom.IsInRing())),
            float(int(hybridization == Chem.rdchem.HybridizationType.SP)),
            float(int(hybridization == Chem.rdchem.HybridizationType.SP2)),
            float(int(hybridization == Chem.rdchem.HybridizationType.SP3)),
        ]
    raise ValueError("feature_variant must be atomic_number, atomic_number_degree, or full")


def bond_feature_vector(bond: Chem.Bond) -> List[float]:
    bond_type = bond.GetBondType()
    stereo = bond.GetStereo()
    return [
        float(bond.GetBondTypeAsDouble()),
        float(int(bond_type == Chem.rdchem.BondType.SINGLE)),
        float(int(bond_type == Chem.rdchem.BondType.DOUBLE)),
        float(int(bond_type == Chem.rdchem.BondType.TRIPLE)),
        float(int(bond_type == Chem.rdchem.BondType.AROMATIC)),
        float(int(bond.GetIsConjugated())),
        float(int(bond.IsInRing())),
        float(int(stereo in {Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOCIS})),
        float(int(stereo in {Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOTRANS})),
        float(int(stereo == Chem.rdchem.BondStereo.STEREOANY)),
    ]


def build_graph_dataset(
    frame: pd.DataFrame,
    feature_variant: str = "full",
):
    configure_runtime()

    import torch
    from torch_geometric.data import Data

    dataset = []
    for _, row in frame.iterrows():
        molecule = Chem.MolFromSmiles(str(row["SMILES"]))
        if molecule is None:
            continue

        node_features = torch.tensor(
            [atom_feature_vector(atom, feature_variant=feature_variant) for atom in molecule.GetAtoms()],
            dtype=torch.float,
        )

        edge_pairs = []  # type: List[List[int]]
        edge_features = []  # type: List[List[float]]
        for bond in molecule.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edge_pairs.extend([[begin, end], [end, begin]])
            bond_features = bond_feature_vector(bond)
            edge_features.extend([bond_features, bond_features])

        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 10), dtype=torch.float)

        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        graph.y = torch.tensor([float(row["Solubility"])], dtype=torch.float)
        graph.global_features = torch.tensor(
            descriptor_vector(molecule=molecule),
            dtype=torch.float,
        )
        graph.name = str(row["Name"])
        graph.smiles = str(row["SMILES"])
        dataset.append(graph)

    return dataset


def save_graph_dataset(dataset, path=GRAPH_DATASET_PATH) -> None:
    configure_runtime()
    import torch

    torch.save(dataset, path)


def split_classical_data(
    X: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify=None,
):
    return train_test_split(
        X,
        y,
        frame,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )


def make_binary_labels(values, threshold: float = -3.0) -> np.ndarray:
    return (np.asarray(values, dtype=float) >= threshold).astype(np.int64)

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import build_classical_feature_matrix, load_dataset, save_classical_arrays


def main() -> None:
    frame = load_dataset()
    X, y, _, _ = build_classical_feature_matrix(frame)
    save_classical_arrays(X, y)
    print(f"Saved classical features with shape {X.shape} and targets with shape {y.shape}.")


if __name__ == "__main__":
    main()

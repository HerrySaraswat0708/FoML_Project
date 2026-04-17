from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.FoML_Project.utils.data_utils import build_graph_dataset, load_dataset, save_graph_dataset


def main() -> None:
    frame = load_dataset()
    dataset = build_graph_dataset(frame)
    save_graph_dataset(dataset)
    print(f"Saved graph dataset with {len(dataset)} molecules.")


if __name__ == "__main__":
    main()

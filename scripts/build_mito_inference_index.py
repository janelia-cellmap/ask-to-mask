"""Build the fixed-FOV mitochondria inference dataset path index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ask_to_mask.training.inference_mito_dataset import (
    build_inference_mito_index,
    _pair_to_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data = config["data"]
    output = args.output or Path(data["index_path"])

    pairs = build_inference_mito_index(
        data_root=data.get("data_root", "/nrs/cellmap/data"),
        em_path_template=data["em_path_template"],
        segmentation_path_template=data["segmentation_path_template"],
        label_name=data.get("label_name", "mito"),
        include_datasets=data.get("include_datasets"),
        skip_datasets=data.get("skip_datasets"),
        raw_target_resolution_nm=data.get("raw_target_resolution_nm"),
        label_target_resolution_nm=data.get("label_target_resolution_nm"),
        require_exact_resolution=data.get("require_exact_resolution", False),
        raw_require_exact_resolution=data.get("raw_require_exact_resolution"),
        label_require_exact_resolution=data.get("label_require_exact_resolution"),
        fov_nm=data.get("fov_nm", 1600.0),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump([_pair_to_json(pair) for pair in pairs], f, indent=2)

    print(f"Wrote {len(pairs)} dataset pairs to {output}")


if __name__ == "__main__":
    main()

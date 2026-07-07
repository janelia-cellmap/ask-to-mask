"""Train the supervised 2.5D EM -> mitochondria mask model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ask_to_mask.training.train_mito_2p5d import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    output_dir = train(args.config, resume_from=args.resume)
    print(f"Supervised 2.5D mitochondria training output: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Submit Flux LoRA training to Janelia LSF with documented GPU-queue defaults."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


GPU_QUEUE_DEFAULTS = {
    "h100": {"queue": "gpu_h100", "slots_per_gpu": 12, "ram_gb_per_slot": 40},
    "h200": {"queue": "gpu_h200", "slots_per_gpu": 12, "ram_gb_per_slot": 40},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--gpu", choices=["h100", "h200"], default="h100")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--mem-gb",
        type=int,
        default=None,
        help="Host memory target in GB; converted to slots using the queue RAM/slot.",
    )
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--threads-per-process", type=int, default=1)
    parser.add_argument("--queue", default=None)
    parser.add_argument("--walltime", default="24:00")
    parser.add_argument("--project", default="cellmap")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/lsf"))
    parser.add_argument(
        "--extra-bsub",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra bsub args appended before the command, e.g. --extra-bsub -R 'select[...]'",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = args.job_name or f"atm_lora_{args.config.stem}"
    log_dir = args.logs_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_defaults = GPU_QUEUE_DEFAULTS[args.gpu]
    queue = args.queue or gpu_defaults["queue"]
    min_slots = gpu_defaults["slots_per_gpu"] * args.num_gpus
    if args.cores is not None:
        min_slots = max(min_slots, args.cores)
    if args.mem_gb is not None:
        mem_slots = -(-args.mem_gb // gpu_defaults["ram_gb_per_slot"])
        min_slots = max(min_slots, mem_slots)
    slots = args.slots or min_slots

    cmd = [
        "bsub",
        "-P",
        args.project,
        "-J",
        job_name,
        "-n",
        str(slots),
        "-q",
        queue,
        "-W",
        args.walltime,
        "-gpu",
        f"num={args.num_gpus}",
        "-o",
        str(log_dir / f"{job_name}.out"),
        "-e",
        str(log_dir / f"{job_name}.err"),
    ]
    cmd.extend(args.extra_bsub)

    train_cmd = (
        "export PYTHONUNBUFFERED=1;"
        " export PYTORCH_ALLOC_CONF=expandable_segments:True;"
        " export TOKENIZERS_PARALLELISM=false;"
        f" export OMP_NUM_THREADS={args.threads_per_process};"
        f" export MKL_NUM_THREADS={args.threads_per_process};"
        f" export OPENBLAS_NUM_THREADS={args.threads_per_process};"
        f" export TBB_NUM_THREADS={args.threads_per_process};"
        f" export OPENMP_NUM_THREADS={args.threads_per_process};"
        f" export NUM_MKL_THREADS={args.threads_per_process};"
        " pixi run train --config " + str(args.config)
    )
    cmd.append(train_cmd)

    print("Submitting LSF job:")
    print(" ".join(cmd))
    print(f"Logs: {log_dir}")
    print(
        f"GPU request: queue={queue}, num_gpus={args.num_gpus}, slots={slots}, "
        f"threads_per_process={args.threads_per_process}"
    )
    if args.dry_run:
        return

    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

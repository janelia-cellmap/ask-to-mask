"""DataLoader `worker_init_fn` for datasets that sample via a stored `self.rng`.

The 2.5D dataset classes (`Mito2p5DInferenceMitoDataset`, `Mito2p5DFixedFovGtDataset`,
`Mito2p5DMixedDataset`, `Mito2p5DSelfSupervisedDataset`) all ignore the `idx`
passed to `__getitem__` and instead draw from a `self.rng` seeded once in
`__init__`, in the main process. With `num_workers > 0` and the default fork
start method, every worker inherits an identical copy of that pre-fork RNG
state -- with no re-seeding, all workers would sample the exact same
sequence, degrading effective sample diversity per batch/epoch.
"""

from __future__ import annotations

import numpy as np
import torch


def reseed_dataset_worker(worker_id: int) -> None:
    """Re-seed every reachable `.rng` in this worker's dataset copy.

    Walks `.pseudo_dataset`/`.gt_dataset` (the attributes `Mito2p5DMixedDataset`
    wraps its sub-datasets under) so both the mixing dataset's own RNG and its
    children's get independent, worker-specific seeds.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    seed_source = np.random.default_rng(int(worker_info.seed % (2**32)))

    visited: set[int] = set()
    stack = [worker_info.dataset]
    while stack:
        obj = stack.pop()
        if id(obj) in visited:
            continue
        visited.add(id(obj))

        rng = getattr(obj, "rng", None)
        if isinstance(rng, np.random.Generator):
            obj.rng = np.random.default_rng(int(seed_source.integers(0, 2**63)))
        elif isinstance(rng, torch.Generator):
            obj.rng = torch.Generator().manual_seed(int(seed_source.integers(0, 2**63)))

        for attr_name in ("pseudo_dataset", "gt_dataset"):
            sub = getattr(obj, attr_name, None)
            if sub is not None:
                stack.append(sub)

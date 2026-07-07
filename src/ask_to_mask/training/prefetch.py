"""CUDA-stream batch prefetcher.

Overlaps the host-to-device transfer of the *next* batch with GPU compute on
the *current* batch, using a side CUDA stream. Without this, `_move_batch_to_device`
still benefits from `pin_memory` + `non_blocking=True`, but the copy is only
issued when the training loop asks for the next batch -- i.e. right when the
GPU would otherwise sit idle waiting for it. This shifts that copy earlier so
it runs concurrently with the previous step's forward/backward pass instead.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

import torch


class CudaPrefetcher:
    """Wraps a batch iterable, moving each batch to `device` one step ahead.

    No-op passthrough (plain iteration, no stream) when `device.type != "cuda"`.
    """

    def __init__(
        self,
        iterable: Iterable[dict[str, Any]],
        device: torch.device,
        move_fn: Callable[[dict[str, Any], torch.device], dict[str, Any]],
    ):
        self._device = device
        self._move_fn = move_fn
        self._iterator: Iterator[dict[str, Any]] = iter(iterable)
        self._stream = torch.cuda.Stream() if device.type == "cuda" else None
        self._next_batch: dict[str, Any] | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._next_batch = None
            return
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                self._next_batch = self._move_fn(batch, self._device)
        else:
            self._next_batch = self._move_fn(batch, self._device)

    def __iter__(self) -> "CudaPrefetcher":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
        batch = self._next_batch
        if batch is None:
            raise StopIteration
        if self._stream is not None:
            # Tensors allocated on the side stream must be marked as used by
            # the current stream, or the caching allocator may reclaim their
            # memory before the consuming kernels actually run.
            for value in batch.values():
                if torch.is_tensor(value):
                    value.record_stream(torch.cuda.current_stream())
                elif isinstance(value, dict):
                    for sub_value in value.values():
                        if torch.is_tensor(sub_value):
                            sub_value.record_stream(torch.cuda.current_stream())
        self._preload()
        return batch

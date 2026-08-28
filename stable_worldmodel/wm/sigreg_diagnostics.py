"""CPU-side collapse diagnostics for post-projection embeddings.

Buffers and metrics never enter the training loss or gradient graph.
Logs ``collapse/{variance_mean,effective_rank_frac,mean_rms}`` to W&B.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import torch

METRIC_NAMES = (
    'variance_mean',
    'effective_rank_frac',
    'mean_rms',
)


@torch.no_grad()
def compute_sigreg_latent_metrics(
    z: torch.Tensor, eps: float = 1e-12
) -> dict[str, float]:
    """Collapse diagnostics on a single time-index buffer.

    z: (N, D) detached float32 CPU embeddings.
    """
    z = z.detach().to(dtype=torch.float32, device='cpu')
    n, dim = z.shape
    if n < 2 or dim < 1:
        return {
            'variance_mean': float('nan'),
            'effective_rank_frac': float('nan'),
            'mean_rms': float('nan'),
        }

    mu = z.mean(dim=0)
    centered = z - mu
    variance_mean = centered.pow(2).mean()
    mean_rms = mu.pow(2).mean().sqrt()

    cov = centered.T @ centered / max(n - 1, 1)
    evals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    probs = evals / (evals.sum() + eps)
    entropy = -(probs * (probs + eps).log()).sum()
    effective_rank_frac = entropy.exp() / dim

    return {
        'variance_mean': float(variance_mean),
        'effective_rank_frac': float(effective_rank_frac),
        'mean_rms': float(mean_rms),
    }


def average_metrics(
    per_time: list[dict[str, float]],
) -> dict[str, float] | None:
    if not per_time:
        return None
    return {
        name: sum(item[name] for item in per_time) / len(per_time)
        for name in METRIC_NAMES
    }


class _CpuFifo:
    """Fixed-capacity CPU ring buffer of embeddings."""

    def __init__(self, capacity: int, dim: int):
        self.data = torch.empty(capacity, dim, dtype=torch.float32)
        self.capacity = capacity
        self.ptr = 0
        self.filled = 0

    def push(self, samples: torch.Tensor) -> None:
        samples = samples.detach().to(dtype=torch.float32, device='cpu')
        n = samples.shape[0]
        if n <= 0:
            return
        if n >= self.capacity:
            self.data.copy_(samples[-self.capacity:])
            self.ptr = 0
            self.filled = self.capacity
            return
        end = self.ptr + n
        if end <= self.capacity:
            self.data[self.ptr : end].copy_(samples)
        else:
            first = self.capacity - self.ptr
            self.data[self.ptr :].copy_(samples[:first])
            self.data[: n - first].copy_(samples[first:])
        self.ptr = end % self.capacity
        self.filled = min(self.capacity, self.filled + n)

    def snapshot(self) -> torch.Tensor:
        if self.filled < self.capacity:
            return self.data[: self.filled].clone()
        return self.data.clone()


class SIGRegCollapseMonitor:
    """Rolling per-time-index CPU buffers and W&B collapse scalar logs."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get('enabled', True))
        self.buffer_size = int(cfg.get('buffer_size', 8192))
        self.sample_every = int(cfg.get('sample_every', 16))
        self.sample_size = int(cfg.get('sample_size', 48))
        self.log_every = int(cfg.get('log_every', 400))
        self.min_samples = int(cfg.get('min_samples', 2048))
        self.eps = float(cfg.get('eps', 1e-12))
        self.background = bool(cfg.get('background', True))

        self._buffers: list[_CpuFifo] | None = None
        self._staging: list[torch.Tensor] = []
        self._pending: tuple[int, torch.Tensor, int] | None = None
        self._staging_i = 0
        self._collect_count = 0
        self._pending_sync: tuple[int, dict[str, float]] | None = None

        self._job_q: queue.Queue | None = None
        self._result_q: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        if self.background:
            self._job_q = queue.Queue(maxsize=1)
            self._result_q = queue.Queue()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name='sigreg-latent-diag',
                daemon=True,
            )
            self._worker.start()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_job_q'] = None
        state['_result_q'] = None
        state['_worker'] = None
        state['_pending'] = None
        state['_staging'] = []
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.background:
            self._job_q = queue.Queue(maxsize=1)
            self._result_q = queue.Queue()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name='sigreg-latent-diag',
                daemon=True,
            )
            self._worker.start()

    def update(self, emb: torch.Tensor, module) -> None:
        if not self.enabled:
            return
        if not self._is_global_zero(module):
            return

        emb = emb.detach()
        step = int(getattr(module, 'global_step', 0))
        self._drain_and_log(module)
        if self.sample_every > 0 and step % self.sample_every == 0:
            self._collect(emb)
        if (
            self.log_every > 0
            and step > 0
            and step % self.log_every == 0
        ):
            self._schedule_compute(step)
            self._drain_and_log(module)

    def close(self, module=None) -> None:
        if module is not None:
            self._drain_and_log(module)
        if self._job_q is not None:
            try:
                self._job_q.put_nowait(None)
            except queue.Full:
                pass

    @staticmethod
    def _is_global_zero(module) -> bool:
        trainer = getattr(module, 'trainer', None)
        if trainer is None:
            return True
        return bool(getattr(trainer, 'is_global_zero', True))

    def _ensure_buffers(self, n_times: int, dim: int, pin: bool) -> None:
        if self._buffers is not None:
            return
        self._buffers = [
            _CpuFifo(self.buffer_size, dim) for _ in range(n_times)
        ]
        try:
            self._staging = [
                torch.empty(
                    self.sample_size,
                    dim,
                    dtype=torch.float32,
                    pin_memory=pin,
                )
                for _ in range(2)
            ]
        except RuntimeError:
            self._staging = [
                torch.empty(self.sample_size, dim, dtype=torch.float32)
                for _ in range(2)
            ]

    def _commit_pending(self) -> None:
        if self._pending is None or self._buffers is None:
            return
        time_idx, staging, n = self._pending
        self._buffers[time_idx].push(staging[:n].clone())
        self._pending = None

    @torch.no_grad()
    def _collect(self, emb: torch.Tensor) -> None:
        if emb.ndim != 3:
            return
        batch, n_times, dim = emb.shape
        if batch < 1 or n_times < 1:
            return

        self._commit_pending()
        self._ensure_buffers(n_times, dim, pin=emb.is_cuda)
        if not self._staging:
            return

        time_idx = self._collect_count % n_times
        self._collect_count += 1
        n_take = min(self.sample_size, batch)
        idx = torch.randperm(batch, device=emb.device)[:n_take]
        sample = emb.detach()[idx, time_idx].to(dtype=torch.float32)

        slot = self._staging[self._staging_i]
        self._staging_i = (self._staging_i + 1) % len(self._staging)
        slot[:n_take].copy_(sample, non_blocking=sample.is_cuda)
        self._pending = (time_idx, slot, n_take)

    def _snapshots(self) -> list[torch.Tensor]:
        self._commit_pending()
        if not self._buffers:
            return []
        return [buf.snapshot() for buf in self._buffers]

    def _schedule_compute(self, step: int) -> None:
        snapshots = self._snapshots()
        if not snapshots:
            return
        if self._job_q is None:
            metrics = self._metrics_from_snapshots(snapshots)
            if metrics is not None:
                self._pending_sync = (step, metrics)
            return
        try:
            self._job_q.put_nowait((step, snapshots))
        except queue.Full:
            return

    def _worker_loop(self) -> None:
        assert self._job_q is not None
        assert self._result_q is not None
        while True:
            job = self._job_q.get()
            if job is None:
                return
            step, snapshots = job
            metrics = self._metrics_from_snapshots(snapshots)
            if metrics is not None:
                self._result_q.put((step, metrics))

    def _metrics_from_snapshots(
        self, snapshots: list[torch.Tensor]
    ) -> dict[str, float] | None:
        if not snapshots:
            return None
        if any(snap.shape[0] < self.min_samples for snap in snapshots):
            return None
        per_time = [
            compute_sigreg_latent_metrics(snap, eps=self.eps)
            for snap in snapshots
        ]
        return average_metrics(per_time)

    def _drain_and_log(self, module) -> None:
        results = []
        if self._pending_sync is not None:
            results.append(self._pending_sync)
            self._pending_sync = None
        if self._result_q is not None:
            while True:
                try:
                    results.append(self._result_q.get_nowait())
                except queue.Empty:
                    break
        for _, metrics in results:
            self._log_scalars(module, metrics)

    def _log_scalars(self, module, metrics: dict[str, float]) -> None:
        payload = {
            f'collapse/{name}': float(metrics[name]) for name in METRIC_NAMES
        }
        module.log_dict(
            payload,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
            sync_dist=False,
        )

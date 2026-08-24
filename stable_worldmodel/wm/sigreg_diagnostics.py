"""CPU-side SIGReg collapse diagnostics for post-projection embeddings.

Buffers and metrics never enter the training loss or gradient graph.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

import torch

METRIC_NAMES = (
    'variance_mean',
    'effective_rank_frac',
    'mean_rms',
)

_TABLE_COLUMNS = (
    'step',
    'metric',
    'value',
    'target',
    'lower_bound',
    'upper_bound',
)

_DEFAULT_BOUNDS = {
    'variance_mean': {
        'target': 1.0,
        'lower_bound': 0.8,
        'upper_bound': 1.2,
    },
    'effective_rank_frac': {
        'target': 1.0,
        'lower_bound': 0.9,
        'upper_bound': 1.0,
    },
    'mean_rms': {
        'target': 0.0,
        'lower_bound': 0.0,
        'upper_bound': 0.05,
    },
}


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


@torch.no_grad()
def calibrate_latent_metric_bounds(
    buffer_size: int = 8192,
    dim: int = 192,
    batch_size: int = 128,
    sample_size: int = 48,
    n_trials: int = 256,
    quantiles: tuple[float, float] = (0.01, 0.99),
    seed: int = 0,
    eps: float = 1e-12,
) -> dict[str, dict[str, float]]:
    """Run the live metric on synthetic N(0, I) with the training sampler.

    Fills each trial buffer by drawing ``sample_size`` rows from batches of
    size ``batch_size``, matching the FIFO collection procedure.
    """
    g = torch.Generator().manual_seed(seed)
    records = {name: [] for name in METRIC_NAMES}
    for _ in range(n_trials):
        chunks = []
        filled = 0
        while filled < buffer_size:
            batch = torch.randn(batch_size, dim, generator=g)
            n_take = min(sample_size, batch_size, buffer_size - filled)
            idx = torch.randperm(batch_size, generator=g)[:n_take]
            chunks.append(batch[idx])
            filled += n_take
        metrics = compute_sigreg_latent_metrics(
            torch.cat(chunks, dim=0), eps=eps
        )
        for name in METRIC_NAMES:
            records[name].append(metrics[name])

    lo_q, hi_q = quantiles
    out = {}
    for name in METRIC_NAMES:
        vals = torch.tensor(records[name], dtype=torch.float32)
        out[name] = {
            'mean': float(vals.mean()),
            'std': float(vals.std(unbiased=False)),
            'q_low': float(torch.quantile(vals, lo_q)),
            'q_high': float(torch.quantile(vals, hi_q)),
            'q01': float(torch.quantile(vals, 0.01)),
            'q99': float(torch.quantile(vals, 0.99)),
        }
    return out


def _merge_bounds(cfg: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    bounds = {
        name: dict(vals) for name, vals in _DEFAULT_BOUNDS.items()
    }
    if not cfg:
        return bounds
    for name in METRIC_NAMES:
        override = cfg.get(name) or {}
        bounds[name].update(
            {
                key: float(override[key])
                for key in ('target', 'lower_bound', 'upper_bound')
                if key in override
            }
        )
    return bounds


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


_SERIES_COLORS = ('#1f77b4', '#d62728', '#7f7f7f', '#7f7f7f')


def _build_diagnostics_figure(rows: list[dict[str, Any]]):
    """Matplotlib overlay: solid value, dotted target, dashed bounds."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n_metrics = len(METRIC_NAMES)
    fig = Figure(
        figsize=(10.0, 2.4 * n_metrics),
        facecolor='white',
        layout='constrained',
    )
    FigureCanvasAgg(fig)
    axes = fig.subplots(n_metrics, 1, sharex=True)
    if n_metrics == 1:
        axes = [axes]

    grouped: dict[str, list[dict[str, Any]]] = {
        name: [] for name in METRIC_NAMES
    }
    for row in rows:
        name = str(row['metric']).rsplit('/', 1)[-1]
        if name in grouped:
            grouped[name].append(row)

    for ax, name in zip(axes, METRIC_NAMES):
        series = sorted(grouped[name], key=lambda row: row['step'])
        if series:
            steps = [row['step'] for row in series]
            ax.plot(
                steps,
                [row['value'] for row in series],
                linestyle='-',
                color=_SERIES_COLORS[0],
                linewidth=2.0,
                label='value',
            )
            ax.plot(
                steps,
                [row['target'] for row in series],
                linestyle=':',
                color=_SERIES_COLORS[1],
                linewidth=1.5,
                label='target',
            )
            ax.plot(
                steps,
                [row['lower_bound'] for row in series],
                linestyle='--',
                color=_SERIES_COLORS[2],
                linewidth=1.25,
                label='lower',
            )
            ax.plot(
                steps,
                [row['upper_bound'] for row in series],
                linestyle='--',
                color=_SERIES_COLORS[3],
                linewidth=1.25,
                label='upper',
            )
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc='best', frameon=False)
    axes[-1].set_xlabel('step')
    fig.suptitle('SIGReg latent collapse diagnostics')
    return fig


class SIGRegCollapseMonitor:
    """Rolling per-time-index CPU buffers and W&B collapse logs."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get('enabled', True))
        self.buffer_size = int(cfg.get('buffer_size', 8192))
        self.sample_every = int(cfg.get('sample_every', 16))
        self.sample_size = int(cfg.get('sample_size', 48))
        self.log_every = int(cfg.get('log_every', 400))
        self.table_log_every = int(cfg.get('table_log_every', 2000))
        self.min_samples = int(cfg.get('min_samples', 2048))
        self.eps = float(cfg.get('eps', 1e-12))
        self.background = bool(cfg.get('background', True))
        self.max_table_rows = int(cfg.get('max_table_rows', 1500))
        self.bounds = _merge_bounds(cfg)

        self._buffers: list[_CpuFifo] | None = None
        self._staging: list[torch.Tensor] = []
        self._pending: tuple[int, torch.Tensor, int] | None = None
        self._staging_i = 0
        self._collect_count = 0
        self._history: list[dict[str, Any]] = []
        self._last_metrics: dict[str, float] | None = None
        self._pending_sync: tuple[int, dict[str, float]] | None = None
        self._last_table_step: int | None = None

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
        for step, metrics in results:
            self._append_history(step, metrics)
            self._log_scalars(module, metrics)
            self._maybe_log_wandb_table(module, step)

    def _append_history(
        self, step: int, metrics: dict[str, float]
    ) -> None:
        self._last_metrics = metrics
        for name in METRIC_NAMES:
            spec = self.bounds[name]
            self._history.append(
                {
                    'step': int(step),
                    'metric': f'collapse/{name}',
                    'value': float(metrics[name]),
                    'target': spec['target'],
                    'lower_bound': spec['lower_bound'],
                    'upper_bound': spec['upper_bound'],
                }
            )
        overflow = len(self._history) - self.max_table_rows
        if overflow > 0:
            self._history = self._history[overflow:]

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

    def _maybe_log_wandb_table(self, module, step: int) -> None:
        if self.table_log_every <= 0 or not self._history:
            return
        if (
            self._last_table_step is not None
            and step - self._last_table_step < self.table_log_every
        ):
            return
        logger = _pl_wandb_logger(module)
        if logger is None or not hasattr(logger, 'log_metrics'):
            return
        try:
            import wandb
        except ImportError:
            return

        table = wandb.Table(
            columns=list(_TABLE_COLUMNS),
            data=[
                [row[col] for col in _TABLE_COLUMNS]
                for row in self._history
            ],
        )
        payload: dict[str, Any] = {'collapse/diagnostics': table}
        try:
            fig = _build_diagnostics_figure(self._history)
            payload['collapse/line'] = wandb.Image(fig)
        except ImportError:
            payload['collapse/line'] = wandb.plot.line(
                table,
                x='step',
                y='value',
                stroke='metric',
                title='SIGReg collapse',
            )
        else:
            try:
                from matplotlib import pyplot as plt

                plt.close(fig)
            except Exception:
                pass
        # Lightning's WandbLogger plots against trainer/global_step and
        # must not be given wandb's step= argument (that drops media).
        logger.log_metrics(payload, step=int(step))
        self._last_table_step = step


def _pl_wandb_logger(module):
    loggers = list(getattr(module, 'loggers', None) or ())
    if not loggers:
        logger = getattr(module, 'logger', None)
        if logger is not None:
            loggers = [logger]
    for logger in loggers:
        if type(logger).__name__ == 'WandbLogger':
            return logger
    return None


if __name__ == '__main__':
    stats = calibrate_latent_metric_bounds()
    print(json.dumps(stats, indent=2))

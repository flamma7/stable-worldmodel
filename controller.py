#!/usr/bin/env python3
"""Dispatch train / mpc / plan jobs from a job_configs YAML.

    python controller.py job_configs/visreg_ogb.yaml
    python controller.py job_configs/visreg_ogb.yaml 0
    python controller.py job_configs/visreg_ogb.yaml 1-5
    python controller.py job_configs/visreg_ogb.yaml train
    python controller.py job_configs/visreg_ogb.yaml mpc --local
    python controller.py job_configs/visreg_ogb.yaml train --at 21:30
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import yaml

import deploy as deploy_mod

HERE = Path(__file__).resolve().parent
JOB_RE = re.compile(r"^job(\d+)$")
RANGE_RE = re.compile(r"^(\d+)-(\d+)$")
MODES = ("train", "mpc", "plan")
SKIP_STATUSES = {"running", "completed"}
TRAIN_SCRIPT = "scripts/train/lewm_visreg.py"
EVAL_SCRIPT = "run_sequential.py"
JOB_META = {
    "mode",
    "params",
    "status",
    "pod_id",
    "pod_ids",
    "completed",
    "result_name",
    "output_model_name",
    "is_hf_model",
}
DEFAULT_PODS_PER_JOB = 2
DEFAULT_PROBE_COUNT = 3
DEFAULT_GPU_ROUNDS = 5
DEFAULT_A_TIMEOUT_S = 120
DEFAULT_B_TIMEOUT_S = 12 * 60
POLL_S = 10
TABLE_S = 15
LAUNCH_INTERVAL_S = 8
POD_STARTUP_DIR = "pod_startup"
TRAIN_SKIP_KEYS = {
    "gpu",
    "region",
    "cloud",
    "num_eval",
    "num_candidates",
    "eval_output_dir",
    "eval_output_hf",
    "save_video",
    "train_script",
    "gpu_options",
    "platform",
    "stack",
    "solver",
    "solver_kwargs",
}

PRINT_LOCK = threading.Lock()
LIVE_PODS = set()
LIVE_LOCK = threading.Lock()
LAUNCH_LOCK = threading.Lock()
_last_launch_at = 0.0


class _IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def log(msg):
    with PRINT_LOCK:
        print(msg, flush=True)


def throttled_launch_direct(announce=None, **kwargs):
    """Call launch_direct with a global 15s gap between successful GPU starts.

    Create errors (sold-out, API failure) do not consume the interval, so the
    next GPU attempt can fire immediately.
    """
    global _last_launch_at
    with LAUNCH_LOCK:
        wait = LAUNCH_INTERVAL_S - (time.monotonic() - _last_launch_at)
        if wait > 0:
            log(f"launch throttle: waiting {wait:.1f}s")
            time.sleep(wait)
        if announce:
            log(announce)
        pod_id = deploy_mod.launch_direct(**kwargs)
        _last_launch_at = time.monotonic()
        return pod_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and deploy or run job commands from a YAML config.",
    )
    parser.add_argument(
        "yaml",
        help="Path to a job_configs YAML (e.g. job_configs/visreg_ogb.yaml)",
    )
    parser.add_argument(
        "select",
        nargs="?",
        default=None,
        help="Job index, inclusive range (1-5), mode (train|mpc|plan), or omit for all",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Run commands here instead of calling deploy.py (gpu=local). "
            "Eval jobs use default.plan.batch_size instead of platform GPU sizes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matched jobs and commands without executing",
    )
    parser.add_argument(
        "--dry-run-container",
        action="store_true",
        help="Deploy the pod with DRY_RUN=1 so startup.sh skips install and waits",
    )
    parser.add_argument(
        "--at",
        default=None,
        metavar="HH:MM",
        help="Local 24-hour time to start (e.g. 21:30). Default: now",
    )
    parser.add_argument(
        "--pods-per-job",
        type=int,
        default=DEFAULT_PODS_PER_JOB,
        metavar="N",
        help="Replica pods to race per platform job (default: 2)",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=DEFAULT_PROBE_COUNT,
        metavar="N",
        help="Max concurrent pods on a GPU type until one reaches B (default: 3)",
    )
    parser.add_argument(
        "--gpu-rounds",
        type=int,
        default=DEFAULT_GPU_ROUNDS,
        metavar="N",
        help="A/B timeouts on a GPU type before skipping it (default: 5)",
    )
    parser.add_argument(
        "--a-timeout",
        type=int,
        default=DEFAULT_A_TIMEOUT_S,
        metavar="SEC",
        help="Seconds to wait for smoke-test heartbeat A (default: 120)",
    )
    parser.add_argument(
        "--b-timeout",
        type=int,
        default=DEFAULT_B_TIMEOUT_S,
        metavar="SEC",
        help="Seconds to wait for install heartbeat B (default: 720)",
    )
    return parser.parse_args()


def parse_at(value):
    raw = str(value).strip()
    if re.fullmatch(r"\d{4}", raw):
        raw = f"{raw[:2]}:{raw[2:]}"
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
    if not match:
        raise SystemExit(f"invalid --at '{value}', expected HH:MM (e.g. 21:30)")
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        raise SystemExit(f"invalid --at '{value}', hour 0-23 and minute 00-59")
    return hour, minute


def wait_until_local(hour, minute):
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    wait_s = (target - now).total_seconds()
    print(
        f"Waiting until {target.strftime('%Y-%m-%d %H:%M')} local "
        f"({int(wait_s)}s / {wait_s / 3600:.1f}h)"
    )
    time.sleep(wait_s)
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} local")


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"yaml not found: {path}")
    return path, load_yaml_dict(path)


def save_yaml(path, cfg):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        yaml.dump(
            cfg,
            f,
            Dumper=_IndentDumper,
            default_flow_style=False,
            sort_keys=False,
        )
    tmp.replace(path)


def load_yaml_dict(path):
    raw = Path(path).read_text()
    cfg = yaml.safe_load(raw)
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path} must be a mapping")
    return cfg


def list_jobs(cfg):
    jobs = []
    for key, value in cfg.items():
        match = JOB_RE.match(str(key))
        if not match:
            continue
        if not isinstance(value, dict):
            raise SystemExit(f"{key} must be a mapping")
        jobs.append((int(match.group(1)), key, value))
    jobs.sort(key=lambda item: item[0])
    if not jobs or jobs[0][0] != 0:
        raise SystemExit("config must define at least job0 (X starting at 0)")
    return jobs


def parse_select(select, jobs):
    """Return (matched job triples, label)."""
    by_index = {index: (index, key, job) for index, key, job in jobs}
    if select is None or select == "" or select == "all":
        return list(jobs), "all"

    if select in MODES:
        matched = [
            item for item in jobs if (item[2].get("mode") or "").lower() == select
        ]
        return matched, f"mode={select}"

    if select.isdigit():
        index = int(select)
        if index not in by_index:
            raise SystemExit(f"no job{index} in config")
        return [by_index[index]], f"job{index}"

    range_match = RANGE_RE.match(select)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if start > end:
            raise SystemExit(f"invalid range {select}")
        missing = [i for i in range(start, end + 1) if i not in by_index]
        if missing:
            raise SystemExit(
                f"range {select} missing job(s): "
                + ", ".join(f"job{i}" for i in missing)
            )
        return [by_index[i] for i in range(start, end + 1)], f"jobs {select}"

    raise SystemExit(
        f"unknown selector '{select}'. "
        "Use a job index, a range like 1-5, or one of: train, mpc, plan"
    )


def job_status(job):
    status = job.get("status")
    if status is None and job.get("completed") is True:
        return "completed"
    if status is None:
        return None
    return str(status).lower()


def should_skip(job):
    return job_status(job) in SKIP_STATUSES


def gpu_key(entry):
    return (
        str(entry.get("gpu") or ""),
        str(entry.get("cloud") or "community").lower(),
    )


class GlobalGpuTracker:
    """Shared probe cap and A/B failure budget per GPU type.

    ``probe_count``: max in-flight pods on an unproven GPU. Extra replicas
    wait. One B marks it healthy and lifts the cap.

    ``gpu_rounds``: A-timeout or B-timeout count. At that many failures the
    GPU is skipped for everyone. Sold-out still skips immediately.
    """

    def __init__(self, gpu_rounds, probe_count):
        self.gpu_rounds = int(gpu_rounds)
        self.probe_count = int(probe_count)
        self._in_flight = {}
        self._fails = {}
        self._skip = set()
        self._healthy = set()
        self._lock = threading.Lock()

    def _score_unlocked(self, key):
        return max(-self.gpu_rounds, self.gpu_rounds - int(self._fails.get(key, 0)))

    def score(self, entry):
        if not entry:
            return 0
        with self._lock:
            return self._score_unlocked(gpu_key(entry))

    def used(self, entry):
        return self.score(entry)

    def fail_count(self, entry):
        if not entry:
            return 0
        with self._lock:
            return int(self._fails.get(gpu_key(entry), 0))

    def _open_unlocked(self, key):
        if key in self._skip:
            return False
        if key in self._healthy:
            return True
        return int(self._in_flight.get(key, 0)) < self.probe_count

    def may_retry(self, platform, start=0):
        """True if some platform GPU from start is not skipped (maybe capped)."""
        start = max(0, int(start or 0))
        with self._lock:
            for i in range(start, len(platform)):
                if gpu_key(platform[i]) not in self._skip:
                    return True
        return False

    def pick(self, platform, start=0):
        start = max(0, int(start or 0))
        with self._lock:
            for i in range(start, len(platform)):
                if self._open_unlocked(gpu_key(platform[i])):
                    return i, platform[i]
        return None, None

    def reserve(self, platform, start=0):
        """Take an in-flight slot. Returns (idx, entry, score) or (None, None, reason)."""
        start = max(0, int(start or 0))
        with self._lock:
            capped = False
            for i in range(start, len(platform)):
                key = gpu_key(platform[i])
                if key in self._skip:
                    continue
                if not self._open_unlocked(key):
                    capped = True
                    continue
                self._in_flight[key] = int(self._in_flight.get(key, 0)) + 1
                return i, platform[i], self._score_unlocked(key)
            return None, None, "capped" if capped else "exhausted"

    def release(self, entry):
        if not entry:
            return
        key = gpu_key(entry)
        with self._lock:
            self._in_flight[key] = max(0, int(self._in_flight.get(key, 0)) - 1)

    def mark_healthy(self, entry):
        if not entry:
            return
        key = gpu_key(entry)
        with self._lock:
            self._healthy.add(key)
            self._skip.discard(key)

    def mark_unavailable(self, entry):
        if not entry:
            return
        key = gpu_key(entry)
        with self._lock:
            self._skip.add(key)

    def record_timeout(self, entry, failures=1):
        """Count an A or B timeout. Skip the GPU once fails >= gpu_rounds."""
        if not entry:
            return -self.gpu_rounds, True
        key = gpu_key(entry)
        with self._lock:
            self._fails[key] = int(self._fails.get(key, 0)) + max(1, int(failures))
            skipped = self._fails[key] >= self.gpu_rounds
            if skipped:
                self._skip.add(key)
                self._healthy.discard(key)
            return self._score_unlocked(key), skipped


def skip_reason(job, key):
    if should_skip(job):
        return f"status={job_status(job)}"
    return None


def merge_params(cfg, job, local=False):
    """default.all <- default.{mode} <- job extras <- job.params, then param_map."""
    mode = (job.get("mode") or "").lower()
    if mode not in MODES:
        raise SystemExit(f"job mode must be one of {MODES}, got {job.get('mode')!r}")

    defaults = cfg.get("default") or {}
    merged = {}
    merged.update(defaults.get("all") or {})
    merged.update(defaults.get(mode) or {})
    for key, value in job.items():
        if key not in JOB_META:
            merged[key] = value
    merged.update(job.get("params") or {})

    param_map = cfg.get("param_map") or {}
    mapped = {}
    for key, value in merged.items():
        mapped[param_map.get(key, key)] = value

    if local:
        mapped["gpu"] = "local"
    return mode, mapped


def format_cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return deploy_mod.format_name_value(value)
    return str(value)


def infer_output_model_name(cfg, job, mapped=None):
    name = job.get("output_model_name")
    if name:
        return name
    parts = [str(cfg.get("name") or "job")]
    for key, value in (job.get("params") or {}).items():
        parts.append(f"{key}{deploy_mod.format_name_value(value)}")
    return "_".join(parts)


def job_sets_batch_size(job):
    return "batch_size" in job or "batch_size" in (job.get("params") or {})


def local_eval_batch_size(cfg):
    """Batch size for --local evals: default.plan.batch_size (50 in tdv_ogb)."""
    plan = (cfg.get("default") or {}).get("plan") or {}
    return plan.get("batch_size", 50)


def dataset_from_mapped(mapped):
    return (
        mapped.get("data.dataset.name")
        or mapped.get("dataset_name")
        or "galilai-group/ogb_cube_single"
    )


def pick(mapped, cfg, key, default=None):
    if key in mapped and mapped[key] is not None:
        return mapped[key]
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


def hf_prefix(cfg, mapped=None):
    mapped = mapped or {}
    prefix = mapped.get("hf.path_prefix") or mapped.get("hf_prefix")
    if prefix:
        return str(prefix)
    name = cfg.get("name")
    if not name:
        raise SystemExit(
            "hf_prefix is required (set it under default.all, or set top-level name)"
        )
    return str(name)


def parse_platform(mapped):
    """Return a list of {gpu, cloud, batch_size} or None if platform is unset."""
    raw = mapped.get("platform")
    if not raw:
        return None
    if isinstance(raw, dict):
        items = []
        for gpu, meta in raw.items():
            entry = dict(meta or {})
            entry.setdefault("gpu", gpu)
            items.append(entry)
        raw = items
    if not isinstance(raw, list) or not raw:
        raise SystemExit("platform must be a non-empty list of GPU mappings")

    default_cloud = mapped.get("cloud") or "community"
    default_batch = mapped.get("loader.batch_size", mapped.get("batch_size"))
    entries = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SystemExit(f"platform[{i}] must be a mapping")
        gpu = item.get("gpu") or item.get("name") or item.get("type")
        if not gpu:
            raise SystemExit(f"platform[{i}] needs gpu")
        cloud = item.get("cloud")
        if not cloud:
            if item.get("secure") is True:
                cloud = "secure"
            elif item.get("community") is True:
                cloud = "community"
            else:
                cloud = default_cloud
        batch_size = item.get("batch_size", default_batch)
        region = item.get("region", mapped.get("region") or "us")
        entries.append(
            {
                "gpu": gpu,
                "cloud": cloud,
                "batch_size": batch_size,
                "region": region,
            }
        )
    return entries


def build_train_cmd(cfg, job, mapped):
    model_name = infer_output_model_name(cfg, job, mapped)
    extras = {k: v for k, v in mapped.items() if k not in TRAIN_SKIP_KEYS}
    extras.pop("output_model_name", None)
    extras["hf.path_prefix"] = hf_prefix(cfg, extras)

    script = pick(mapped, cfg, "train_script", TRAIN_SCRIPT)
    parts = [f"python {script}", f"output_model_name={model_name}"]
    for key, value in extras.items():
        parts.append(f"{key}={format_cli_value(value)}")
    return " ".join(parts), model_name


def build_eval_cmd(cfg, job, mapped, mode, local=False):
    model_name = infer_output_model_name(cfg, job, mapped)
    seed = mapped.get("seed", 42)
    num_eval = mapped.get("num_eval", 50)
    batch_size = mapped.get("loader.batch_size", mapped.get("batch_size", 50))
    repo = pick(mapped, cfg, "hf.repo_id") or pick(mapped, cfg, "hf_repo")
    if not repo:
        raise SystemExit("hf_repo is required (set it under default.all or at the top level)")
    parts = [
        f"python {EVAL_SCRIPT}",
        mode,
        shlex.quote(str(model_name)),
        str(seed),
        str(num_eval),
        "--batch-size",
        str(batch_size),
        "--hf-repo",
        str(repo),
        "--hf-subdir",
        hf_prefix(cfg, mapped),
        "--eval-output-dir",
        str(pick(mapped, cfg, "eval_output_dir", "data")),
        "--dataset",
        str(dataset_from_mapped(mapped)),
    ]
    if mode == "plan":
        parts.extend(
            ["--num-candidates", str(mapped.get("num_candidates", 64))]
        )
    if mode == "mpc":
        parts.extend(["--solver", str(mapped.get("solver", "icem"))])
        solver_kwargs = mapped.get("solver_kwargs") or {}
        if isinstance(solver_kwargs, dict):
            for key, value in solver_kwargs.items():
                parts.extend(
                    ["--solver-kw", f"{key}={format_cli_value(value)}"]
                )
    if job.get("is_hf_model"):
        parts.append("--is-hf-model")
    if local:
        parts.append("--local")
    return " ".join(parts), model_name


def build_job(cfg, job, local=False):
    mode, mapped = merge_params(cfg, job, local=local)
    platform = parse_platform(mapped)
    use_replicas = bool(platform) and not local
    if platform:
        first = platform[0]
        mapped = dict(mapped)
        mapped["gpu"] = first["gpu"] if not local else "local"
        mapped["cloud"] = first["cloud"]
        if first.get("region"):
            mapped["region"] = first["region"]
        if not local and first.get("batch_size") is not None:
            mapped["loader.batch_size"] = first["batch_size"]
            mapped["batch_size"] = first["batch_size"]
        gpu = mapped["gpu"]
    else:
        gpu = mapped.get("gpu")
        if not gpu:
            raise SystemExit(
                f"no gpu for mode={mode} (set default.{mode}.gpu, platform, or --local)"
            )
        platform = [
            {
                "gpu": gpu,
                "cloud": mapped.get("cloud") or "community",
                "batch_size": mapped.get("loader.batch_size", mapped.get("batch_size")),
                "region": mapped.get("region") or "us",
            }
        ]
    if local and mode != "train" and not job_sets_batch_size(job):
        mapped = dict(mapped)
        bs = local_eval_batch_size(cfg)
        mapped["loader.batch_size"] = bs
        mapped["batch_size"] = bs
    if mode == "train":
        cmd, model_name = build_train_cmd(cfg, job, mapped)
        install_mode = "train"
    else:
        cmd, model_name = build_eval_cmd(cfg, job, mapped, mode, local=local)
        install_mode = "eval"
    repo = pick(mapped, cfg, "hf.repo_id") or pick(mapped, cfg, "hf_repo")
    return {
        "mode": mode,
        "install_mode": install_mode,
        "cmd": cmd,
        "gpu": gpu,
        "region": mapped.get("region") or "us",
        "cloud": mapped.get("cloud") or "community",
        "model_name": model_name,
        "mapped": mapped,
        "dataset": dataset_from_mapped(mapped),
        "platform": platform,
        "use_replicas": use_replicas,
        "hf_repo": repo,
        "hf_subdir": hf_prefix(cfg, mapped),
    }


def apply_gpu_entry(cfg, key, spec, gpu_entry):
    job = cfg[key]
    mapped = dict(spec["mapped"])
    mapped["gpu"] = gpu_entry["gpu"]
    mapped["cloud"] = gpu_entry["cloud"]
    if gpu_entry.get("region"):
        mapped["region"] = gpu_entry["region"]
    if gpu_entry.get("batch_size") is not None:
        mapped["loader.batch_size"] = gpu_entry["batch_size"]
        mapped["batch_size"] = gpu_entry["batch_size"]
    mode = spec["mode"]
    if mode == "train":
        cmd, model_name = build_train_cmd(cfg, job, mapped)
        install_mode = "train"
    else:
        cmd, model_name = build_eval_cmd(cfg, job, mapped, mode)
        install_mode = "eval"
    new = dict(spec)
    new["mapped"] = mapped
    new["gpu"] = gpu_entry["gpu"]
    new["cloud"] = gpu_entry["cloud"]
    if gpu_entry.get("region"):
        new["region"] = gpu_entry["region"]
    new["cmd"] = cmd
    new["model_name"] = model_name
    new["install_mode"] = install_mode
    return new


def stack_size(spec):
    if spec.get("use_replicas"):
        return 1
    if spec["mode"] == "train":
        return 1
    raw = spec["mapped"].get("stack", 1)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise SystemExit(f"stack must be an integer, got {raw!r}")
    if n < 1:
        raise SystemExit(f"stack must be >= 1, got {n}")
    return n


def stack_signature(spec, n):
    return (
        spec["mode"],
        spec["gpu"],
        spec["region"],
        spec["cloud"],
        spec["dataset"],
        n,
    )


def stack_jobs(to_run):
    """Pack consecutive compatible mpc/plan jobs into groups of size `stack`."""
    groups = []
    current = []
    current_sig = None
    current_n = 1
    for item in to_run:
        spec = item[1]
        n = stack_size(spec)
        if n <= 1:
            if current:
                groups.append(current)
                current = []
                current_sig = None
            groups.append([item])
            continue
        sig = stack_signature(spec, n)
        if current and (sig != current_sig or len(current) >= current_n):
            groups.append(current)
            current = []
        if not current:
            current_sig = sig
            current_n = n
        current.append(item)
    if current:
        groups.append(current)
    return groups


def short_gpu(name):
    return (
        str(name)
        .replace("NVIDIA GeForce ", "")
        .replace("NVIDIA RTX ", "RTX ")
        .replace("NVIDIA ", "")
        .replace(" Generation", "")
    )


def format_age(seconds):
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{sec:02d}s"


def print_plan(matched, skipped, groups, local, args):
    print("Matched jobs: " + ", ".join(key for _, key, _ in matched))
    if skipped:
        reasons = []
        for _, key, job in skipped:
            reason = skip_reason(job, key) or f"status={job_status(job)}"
            reasons.append(f"{key} ({reason})")
        print("Skipped: " + ", ".join(reasons))
    else:
        print("Skipped: none")
    if not groups:
        print("Nothing to run.")
        return
    dest = "locally" if local else "in parallel via deploy.py"
    n_jobs = sum(len(group) for group in groups)
    unit = "run" if local else "job group"
    print(f"Will run {n_jobs} job(s) in {len(groups)} {unit}(s) {dest}:")
    for group in groups:
        keys = [key for key, _ in group]
        spec = group[0][1]
        label = "+".join(keys)
        replicas = args.pods_per_job if spec.get("use_replicas") else 1
        print(
            f"  {label}  mode={spec['mode']}  region={spec['region']}  "
            f"replicas={replicas}  stack={len(group)}"
        )
        if spec.get("use_replicas"):
            print(
                f"    probes={args.probe_count}  gpu-rounds={args.gpu_rounds}  "
                f"A={args.a_timeout}s  B={args.b_timeout}s"
            )
            for i, entry in enumerate(spec["platform"], start=1):
                print(
                    f"    {i}. {entry['gpu']}  cloud={entry['cloud']}  "
                    f"batch_size={entry['batch_size']}"
                )
        else:
            print(
                f"    gpu={spec['gpu']}  cloud={spec['cloud']}"
            )
        for i, (_, item) in enumerate(group):
            print(f"    CMD_{i}: {item['cmd']}")


def dataset_dir(dataset):
    return str(dataset).replace("/", "--")


def register_pod(pod_id):
    if not pod_id:
        return
    with LIVE_LOCK:
        LIVE_PODS.add(pod_id)


def unregister_pod(pod_id):
    if not pod_id:
        return
    with LIVE_LOCK:
        LIVE_PODS.discard(pod_id)


def kill_registered(pod_id, reason=""):
    if not pod_id:
        return
    suffix = f" ({reason})" if reason else ""
    log(f"killing {pod_id}{suffix}")
    try:
        deploy_mod.kill_pod(pod_id)
    except Exception as exc:
        log(f"  kill {pod_id} failed: {exc}")
    unregister_pod(pod_id)


def kill_all_live():
    with LIVE_LOCK:
        ids = list(LIVE_PODS)
    for pod_id in ids:
        kill_registered(pod_id, "controller exit")


def persist_group(yaml_path, cfg, yaml_lock, group, status, pod_id=None, pod_ids=None):
    """Patch only this group's jobs on disk. Never rewrite the whole in-memory cfg."""
    with yaml_lock:
        on_disk = load_yaml_dict(yaml_path)
        for key, spec in group:
            job = on_disk.get(key)
            if not isinstance(job, dict):
                job = dict(cfg.get(key) or {})
                on_disk[key] = job
            job["status"] = status
            job["output_model_name"] = spec["model_name"]
            if pod_id:
                job["pod_id"] = pod_id
            if pod_ids is not None:
                job["pod_ids"] = list(pod_ids)
            live = cfg.get(key)
            if isinstance(live, dict):
                live.update(job)
        save_yaml(yaml_path, on_disk)


def persist_status(ctx, group, status, slots=None, pod_id=None, pod_ids=None):
    live_ids = [slot.pod_id for slot in (slots or []) if slot.pod_id]
    if pod_ids is None:
        pod_ids = live_ids
    winner = next(
        (slot.pod_id for slot in (slots or []) if slot.note == "winner" and slot.pod_id),
        None,
    )
    if winner:
        pod_id = winner
    persist_group(
        ctx["yaml_path"],
        ctx["cfg"],
        ctx["yaml_lock"],
        group,
        status,
        pod_id=pod_id,
        pod_ids=pod_ids,
    )


def ensure_pod_startup_dir(repo):
    """Create pod_startup/ on the HF model repo if it is missing."""
    from huggingface_hub import HfApi

    api = HfApi()
    keep = f"{POD_STARTUP_DIR}/.gitkeep"
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except Exception as exc:
        raise RuntimeError(f"could not list {repo}: {exc}") from exc
    prefix = POD_STARTUP_DIR + "/"
    if any(path == keep or path.startswith(prefix) for path in files):
        log(f"HF heartbeat dir exists: {repo}/{POD_STARTUP_DIR}")
        return
    log(f"creating {repo}/{POD_STARTUP_DIR}")
    api.upload_file(
        path_or_fileobj=b"",
        path_in_repo=keep,
        repo_id=repo,
        repo_type="model",
        commit_message=f"create {POD_STARTUP_DIR}",
    )


class HeartbeatIndex:
    def __init__(self, repo, poll_s=POLL_S):
        self.repo = repo
        self.folder = POD_STARTUP_DIR
        self.poll_s = poll_s
        self._names = set()
        self._lock = threading.Lock()
        self._last = 0.0

    def refresh(self, force=False):
        now = time.time()
        with self._lock:
            if not force and now - self._last < self.poll_s:
                return
            self._last = now
        names = set()
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            if hasattr(api, "list_repo_tree"):
                items = api.list_repo_tree(
                    repo_id=self.repo,
                    repo_type="model",
                    path_in_repo=self.folder,
                    recursive=False,
                )
                for item in items:
                    path = getattr(item, "path", None) or str(item)
                    names.add(path.rsplit("/", 1)[-1])
            else:
                files = api.list_repo_files(repo_id=self.repo, repo_type="model")
                prefix = self.folder + "/"
                names = {
                    path.rsplit("/", 1)[-1]
                    for path in files
                    if path.startswith(prefix)
                }
        except Exception as exc:
            log(f"heartbeat list failed ({self.repo}/{self.folder}): {exc}")
            return
        with self._lock:
            self._names = names

    def stage(self, pod_id):
        if not pod_id:
            return None
        self.refresh()
        with self._lock:
            if f"{pod_id}-B.txt" in self._names:
                return "B"
            if f"{pod_id}-A.txt" in self._names:
                return "A"
        return None


class StatusBoard:
    def __init__(self):
        self._lock = threading.Lock()
        self._rows = {}
        self._stop = threading.Event()

    def update(self, key, **fields):
        with self._lock:
            row = self._rows.setdefault(key, {"key": key})
            row.update(fields)

    def stop(self):
        self._stop.set()

    def render(self):
        with self._lock:
            rows = [dict(row) for row in self._rows.values()]
        rows.sort(key=lambda row: (str(row.get("job") or ""), int(row.get("replica") or 0)))
        now = time.time()
        header = (
            f"{'job':<16} {'r':>2} {'gpu':<22} {'cloud':<10} {'pod':<16} "
            f"{'hb':<2} {'att':>3} {'rnd':>3} {'age':>7}  note"
        )
        lines = [
            "",
            f"=== {datetime.now().strftime('%H:%M:%S')} job status ===",
            header,
            "-" * len(header),
        ]
        if not rows:
            lines.append("(no jobs yet)")
        for row in rows:
            launched = row.get("launched_at")
            age = format_age(now - launched) if launched else "-"
            lines.append(
                f"{str(row.get('job') or '-'):<16} "
                f"{int(row.get('replica') or 0):>2} "
                f"{str(row.get('gpu') or '-'):<22} "
                f"{str(row.get('cloud') or '-'):<10} "
                f"{str(row.get('pod') or '-'):<16} "
                f"{str(row.get('hb') or '-'):<2} "
                f"{int(row.get('attempt') or 0):>3} "
                f"{int(row.get('round') or 0):>3} "
                f"{age:>7}  "
                f"{row.get('note') or ''}"
            )
        log("\n".join(lines))

    def loop(self):
        while not self._stop.wait(TABLE_S):
            self.render()


def launch_group_pod(cfg, group, gpu_entry, replica, dry_run_container=False):
    keys = [key for key, _ in group]
    specs = [apply_gpu_entry(cfg, key, spec, gpu_entry) for key, spec in group]
    spec = specs[0]
    label = "+".join(keys)
    env = {
        "MODE": spec["install_mode"],
        "HF_DATASET": spec["dataset"],
        "HF_DATASET_DIR": dataset_dir(spec["dataset"]),
        "OUTPUT_MODEL_NAME": spec["model_name"],
        "HF_REPO": spec["hf_repo"] or "",
        "HF_SUBDIR": spec["hf_subdir"],
    }
    for i, item in enumerate(specs):
        env[f"CMD_{i}"] = item["cmd"]
    if dry_run_container:
        env["DRY_RUN"] = "1"
    key_part = "_".join(keys)
    seed = spec.get("mapped", {}).get("seed", 42)
    seed_part = f"_s{seed}" if spec["mode"] in ("mpc", "plan") else ""
    pod_name = (
        f"{spec['model_name']}_{spec['mode']}_{key_part}{seed_part}_r{replica}".replace(
            "/", "-"
        )
    )
    pod_id = throttled_launch_direct(
        announce=(
            f"{label} r{replica}: launching {pod_name} on {spec['gpu']} "
            f"({spec['cloud']}, {spec['region']}) batch_size="
            f"{gpu_entry.get('batch_size')}"
        ),
        name=pod_name,
        gpu=spec["gpu"],
        extra_env=env,
        region=spec["region"],
        cloud=spec["cloud"],
        wait=0,
    )
    register_pod(pod_id)
    return pod_id, specs


class ReplicaSlot:
    def __init__(self, index):
        self.index = index
        self.pod_id = None
        self.launched_at = None
        self.attempt = 0
        self.stage = None
        self.note = "queued"
        self.dead = False
        self.gpu_index = 0
        self.gpu_entry = None

    def reset_pod(self):
        self.pod_id = None
        self.launched_at = None
        self.stage = None


def replica_count(spec, args):
    if spec.get("use_replicas"):
        n = int(args.pods_per_job)
        if n < 1:
            raise SystemExit("--pods-per-job must be >= 1")
        return n
    return 1


def run_replicas(ctx, group, n_replicas):
    args = ctx["args"]
    board = ctx["board"]
    heartbeats = ctx["heartbeats"]
    tracker = ctx["gpu_tracker"]
    label = ctx["label"]
    platform = list(group[0][1]["platform"])
    slots = [ReplicaSlot(i) for i in range(n_replicas)]

    def row_key(slot):
        return f"{label}:r{slot.index}"

    def entry_for(slot):
        return slot.gpu_entry or {}

    def paint(slot, **fields):
        entry = entry_for(slot)
        gpu_score = tracker.score(entry) if entry else slot.attempt
        payload = {
            "job": label,
            "replica": slot.index,
            "gpu": short_gpu(entry.get("gpu")) if entry.get("gpu") else "-",
            "cloud": entry.get("cloud") or "-",
            "pod": slot.pod_id or "-",
            "hb": slot.stage or "-",
            "attempt": gpu_score,
            "round": gpu_score,
            "launched_at": slot.launched_at,
            "note": slot.note,
        }
        payload.update(fields)
        board.update(row_key(slot), **payload)

    def flush(status="deploying"):
        persist_status(ctx, group, status, slots=slots)

    def can_relaunch(slot):
        return tracker.may_retry(platform, start=slot.gpu_index)

    def launch_slot(slot, depth=0):
        if depth > len(platform) + 1:
            slot.dead = True
            slot.note = "all GPUs exhausted"
            paint(slot)
            flush()
            return False
        idx, entry, gpu_score = tracker.reserve(platform, start=slot.gpu_index)
        if entry is None:
            slot.dead = gpu_score != "capped"
            slot.note = (
                "waiting for probe slot"
                if gpu_score == "capped"
                else "all GPUs exhausted"
            )
            paint(slot)
            flush()
            return False
        slot.gpu_index = idx
        slot.gpu_entry = entry
        slot.attempt += 1
        slot.dead = False
        slot.reset_pod()
        slot.note = (
            f"launching {short_gpu(entry['gpu'])} "
            f"(rounds left {gpu_score}/{tracker.gpu_rounds})"
        )
        paint(slot)
        flush()
        try:
            pod_id, _ = launch_group_pod(
                ctx["cfg"],
                group,
                entry,
                slot.index,
                dry_run_container=args.dry_run_container,
            )
        except deploy_mod.NoGpuAvailable as exc:
            log(
                f"{label} r{slot.index}: {short_gpu(entry['gpu'])} unavailable; "
                f"moving on: {exc}"
            )
            tracker.release(entry)
            tracker.mark_unavailable(entry)
            slot.gpu_index = idx + 1
            slot.note = f"{short_gpu(entry['gpu'])} unavailable"
            paint(slot)
            flush()
            return launch_slot(slot, depth=depth + 1)
        except Exception as exc:
            if deploy_mod.is_no_gpu_error(str(exc)):
                log(
                    f"{label} r{slot.index}: {short_gpu(entry['gpu'])} unavailable; "
                    f"moving on: {exc}"
                )
                tracker.release(entry)
                tracker.mark_unavailable(entry)
                slot.gpu_index = idx + 1
                slot.note = f"{short_gpu(entry['gpu'])} unavailable"
                paint(slot)
                flush()
                return launch_slot(slot, depth=depth + 1)
            tracker.release(entry)
            log(f"{label} r{slot.index}: launch failed: {exc}")
            slot.note = f"launch failed: {exc}; retrying"
            paint(slot)
            flush()
            return False
        slot.pod_id = pod_id
        slot.launched_at = time.time()
        slot.stage = None
        slot.note = "waiting A"
        paint(slot)
        flush()
        return True

    def recycle_slot(slot, reason):
        entry = slot.gpu_entry
        if slot.pod_id:
            kill_registered(slot.pod_id, f"{label} r{slot.index} {reason}")
            slot.pod_id = None
        if entry:
            tracker.release(entry)
        if reason in {"A timeout", "B timeout", "pod dead"} and entry:
            score, skipped = tracker.record_timeout(entry)
            fails = tracker.fail_count(entry)
            if skipped:
                slot.gpu_index = (slot.gpu_index or 0) + 1
                log(
                    f"{label} r{slot.index}: {short_gpu(entry.get('gpu'))} {reason} "
                    f"({fails}/{tracker.gpu_rounds}); skipping to next GPU"
                )
            else:
                log(
                    f"{label} r{slot.index}: {short_gpu(entry.get('gpu'))} {reason} "
                    f"({fails}/{tracker.gpu_rounds}); rounds left {score}"
                )
        slot.reset_pod()
        slot.note = reason
        paint(slot)
        flush()
        if can_relaunch(slot):
            log(f"{label} r{slot.index}: {reason}; relaunching")
            return launch_slot(slot)
        slot.dead = True
        slot.note = f"{reason}; all GPUs exhausted"
        paint(slot)
        flush()
        return False

    if not any(slot.pod_id and not slot.dead for slot in slots):
        for slot in slots:
            launch_slot(slot)

    while True:
        now = time.time()
        winner = None
        stage_changed = False
        for slot in slots:
            if slot.dead or not slot.pod_id:
                continue
            active = deploy_mod.is_pod_active(slot.pod_id)
            if active is False:
                log(
                    f"{label} r{slot.index}: pod {slot.pod_id} is not running; "
                    "recycling"
                )
                recycle_slot(slot, "pod dead")
        for slot in slots:
            if slot.dead or not slot.pod_id:
                continue
            stage = heartbeats.stage(slot.pod_id)
            if stage and stage != slot.stage:
                slot.stage = stage
                slot.note = "waiting B" if stage == "A" else "install ok"
                paint(slot)
                stage_changed = True
            if slot.stage == "B":
                winner = slot
                break
        if stage_changed and not winner:
            flush()

        if winner:
            tracker.mark_healthy(winner.gpu_entry)
            tracker.release(winner.gpu_entry)
            for slot in slots:
                if slot is winner:
                    slot.note = "winner"
                    paint(slot, hb="B")
                    unregister_pod(winner.pod_id)
                    continue
                if slot.pod_id:
                    kill_registered(slot.pod_id, f"{label} sibling reached B")
                    slot.pod_id = None
                    tracker.release(slot.gpu_entry)
                slot.dead = True
                slot.note = "killed (sibling B)"
                paint(slot)
            flush("running")
            return winner

        for slot in slots:
            if slot.dead:
                if can_relaunch(slot):
                    slot.dead = False
                    launch_slot(slot)
                continue
            if not slot.pod_id:
                if can_relaunch(slot):
                    launch_slot(slot)
                else:
                    slot.dead = True
                    slot.note = slot.note or "all GPUs exhausted"
                    paint(slot)
                continue
            if not slot.launched_at:
                continue
            elapsed = now - slot.launched_at
            if slot.stage is None and elapsed >= args.a_timeout:
                recycle_slot(slot, "A timeout")
                continue
            if slot.stage == "A" and elapsed >= args.b_timeout:
                recycle_slot(slot, "B timeout")

        waiting = any(slot.pod_id and not slot.dead for slot in slots)
        if waiting or any(can_relaunch(slot) for slot in slots):
            time.sleep(POLL_S)
            continue
        flush()
        return None


def run_group_remote(ctx, group):
    spec = group[0][1]
    label = ctx["label"]
    n_replicas = replica_count(spec, ctx["args"])

    result = run_replicas(ctx, group, n_replicas)
    if isinstance(result, ReplicaSlot):
        persist_status(
            ctx,
            group,
            "running",
            slots=[result],
            pod_id=result.pod_id,
            pod_ids=[result.pod_id],
        )
        log(f"{label}: B on {result.pod_id}; wrote status=running")
        return True
    persist_status(ctx, group, "failed")
    log(f"{label}: all GPUs exhausted")
    return False

def run_local(group):
    env = os.environ.copy()
    env.setdefault("STABLEWM_HOME", str(HERE))
    for i, (key, spec) in enumerate(group):
        print(f"{key}: running locally (CMD_{i})")
        print(f"  {spec['cmd']}")
        subprocess.run(
            spec["cmd"],
            shell=True,
            check=True,
            cwd=HERE,
            env=env,
        )


def run_all_parallel(groups, cfg, yaml_path, args):
    yaml_lock = threading.Lock()
    board = StatusBoard()
    indexes = {}
    if args.gpu_rounds < 1:
        raise SystemExit("--gpu-rounds must be >= 1")
    if args.probe_count < 1:
        raise SystemExit("--probe-count must be >= 1")
    gpu_tracker = GlobalGpuTracker(args.gpu_rounds, args.probe_count)
    for group in groups:
        spec = group[0][1]
        repo = spec.get("hf_repo")
        if not repo:
            raise SystemExit("hf_repo is required for pod heartbeats")
        if repo not in indexes:
            ensure_pod_startup_dir(repo)
            indexes[repo] = HeartbeatIndex(repo)
        for i in range(replica_count(spec, args)):
            board.update(
                f"{'+'.join(k for k, _ in group)}:r{i}",
                job="+".join(k for k, _ in group),
                replica=i,
                gpu="-",
                cloud="-",
                pod="-",
                hb="-",
                attempt=0,
                round=0,
                note="queued",
            )

    printer = threading.Thread(target=board.loop, name="status-table", daemon=True)
    printer.start()
    board.render()

    def worker(group):
        keys = [key for key, _ in group]
        label = "+".join(keys)
        spec = group[0][1]
        ctx = {
            "cfg": cfg,
            "yaml_path": yaml_path,
            "yaml_lock": yaml_lock,
            "args": args,
            "board": board,
            "heartbeats": indexes[spec["hf_repo"]],
            "label": label,
            "gpu_tracker": gpu_tracker,
        }
        return label, run_group_remote(ctx, group)

    try:
        with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
            futs = [pool.submit(worker, group) for group in groups]
            for fut in as_completed(futs):
                label, ok = fut.result()
                log(f"{label}: {'ready' if ok else 'failed'}")
    finally:
        board.stop()
        printer.join(timeout=1)
        board.render()


def main():
    args = parse_args()
    yaml_path, cfg = load_yaml(args.yaml)
    jobs = list_jobs(cfg)
    matched, label = parse_select(args.select, jobs)
    print(f"Selector: {label}  ({yaml_path})")

    skipped = []
    runnable = []
    for item in matched:
        _, key, job = item
        if skip_reason(job, key):
            skipped.append(item)
        else:
            runnable.append(item)

    to_run = []
    for _, key, job in runnable:
        to_run.append((key, build_job(cfg, job, local=args.local)))

    groups = stack_jobs(to_run)
    print_plan(matched, skipped, groups, args.local, args)
    if args.dry_run or not groups:
        return

    if args.at:
        hour, minute = parse_at(args.at)
        wait_until_local(hour, minute)

    if args.local:
        for group in groups:
            keys = [key for key, _ in group]
            label = "+".join(keys)
            run_local(group)
            persist_group(
                yaml_path,
                cfg,
                threading.Lock(),
                group,
                "completed",
            )
            print(f"{label}: wrote status=completed to {yaml_path}")
        return

    try:
        run_all_parallel(groups, cfg, yaml_path, args)
    except KeyboardInterrupt:
        log("interrupted; terminating live pods")
        kill_all_live()
        raise


if __name__ == "__main__":
    main()

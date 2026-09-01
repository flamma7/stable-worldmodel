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
import time
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
    "completed",
    "result_name",
    "output_model_name",
    "is_hf_model",
}
DEPLOY_ATTEMPTS = 5
DEPLOY_WAIT_S = 60
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
    "stack",
}


class _IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


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
        help="Run commands here instead of calling deploy.py (gpu=local)",
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
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path} must be a mapping")
    return path, cfg


def save_yaml(path, cfg):
    with Path(path).open("w") as f:
        yaml.dump(
            cfg,
            f,
            Dumper=_IndentDumper,
            default_flow_style=False,
            sort_keys=False,
        )


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


def build_eval_cmd(cfg, job, mapped, mode):
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
    if job.get("is_hf_model"):
        parts.append("--is-hf-model")
    return " ".join(parts), model_name


def build_job(cfg, job, local=False):
    mode, mapped = merge_params(cfg, job, local=local)
    if mode == "train":
        cmd, model_name = build_train_cmd(cfg, job, mapped)
        install_mode = "train"
    else:
        cmd, model_name = build_eval_cmd(cfg, job, mapped, mode)
        install_mode = "eval"
    gpu = mapped.get("gpu")
    if not gpu:
        raise SystemExit(f"no gpu for mode={mode} (set default.{mode}.gpu or --local)")
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
    }


def stack_size(spec):
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


def print_plan(matched, skipped, groups, local):
    print("Matched jobs: " + ", ".join(key for _, key, _ in matched))
    if skipped:
        reasons = []
        for _, key, job in skipped:
            reasons.append(f"{key} (status={job_status(job)})")
        print("Skipped: " + ", ".join(reasons))
    else:
        print("Skipped: none")
    if not groups:
        print("Nothing to run.")
        return
    dest = "locally" if local else "via deploy.py"
    n_jobs = sum(len(group) for group in groups)
    unit = "run" if local else "pod"
    print(f"Will run {n_jobs} job(s) in {len(groups)} {unit}(s) {dest}:")
    for group in groups:
        keys = [key for key, _ in group]
        spec = group[0][1]
        label = "+".join(keys)
        print(
            f"  {label}  mode={spec['mode']}  gpu={spec['gpu']}  "
            f"cloud={spec['cloud']}  region={spec['region']}  "
            f"stack={len(group)}"
        )
        for i, (_, item) in enumerate(group):
            print(f"    CMD_{i}: {item['cmd']}")


def dataset_dir(dataset):
    return str(dataset).replace("/", "--")


def deploy_job(group, dry_run_container=False):
    keys = [key for key, _ in group]
    spec = group[0][1]
    label = "+".join(keys)
    env = {
        "MODE": spec["install_mode"],
        "HF_DATASET": spec["dataset"],
        "HF_DATASET_DIR": dataset_dir(spec["dataset"]),
        "OUTPUT_MODEL_NAME": spec["model_name"],
    }
    for i, (_, item) in enumerate(group):
        env[f"CMD_{i}"] = item["cmd"]
    if dry_run_container:
        env["DRY_RUN"] = "1"
    key_part = "_".join(keys)
    pod_name = f"{spec['model_name']}_{spec['mode']}_{key_part}".replace("/", "-")
    print(
        f"{label}: deploying {pod_name} on {spec['gpu']} "
        f"({spec['cloud']}, {spec['region']})"
    )
    if dry_run_container:
        print("  DRY_RUN=1 (container will skip install and wait)")
    last_error = None
    for attempt in range(1, DEPLOY_ATTEMPTS + 1):
        print(f"{label}: deploy attempt {attempt}/{DEPLOY_ATTEMPTS}")
        try:
            return deploy_mod.launch_direct(
                name=pod_name,
                gpu=spec["gpu"],
                extra_env=env,
                region=spec["region"],
                cloud=spec["cloud"],
                wait=DEPLOY_WAIT_S,
            )
        except RuntimeError as exc:
            last_error = exc
            print(f"{label}: attempt {attempt} failed: {exc}")
            if attempt < DEPLOY_ATTEMPTS:
                print(f"{label}: retrying (install/pod died within {DEPLOY_WAIT_S}s)")
    raise RuntimeError(
        f"{label}: deploy failed after {DEPLOY_ATTEMPTS} attempts: {last_error}"
    )


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


def main():
    args = parse_args()
    yaml_path, cfg = load_yaml(args.yaml)
    jobs = list_jobs(cfg)
    matched, label = parse_select(args.select, jobs)
    print(f"Selector: {label}  ({yaml_path})")

    skipped = [(i, key, job) for i, key, job in matched if should_skip(job)]
    runnable = [(i, key, job) for i, key, job in matched if not should_skip(job)]

    to_run = []
    for _, key, job in runnable:
        to_run.append((key, build_job(cfg, job, local=args.local)))

    groups = stack_jobs(to_run)
    print_plan(matched, skipped, groups, args.local)
    if args.dry_run or not groups:
        return

    if args.at:
        hour, minute = parse_at(args.at)
        wait_until_local(hour, minute)

    for group in groups:
        keys = [key for key, _ in group]
        label = "+".join(keys)
        if args.local:
            run_local(group)
            status = "completed"
            for key, spec in group:
                job = cfg[key]
                job["status"] = status
                job["output_model_name"] = spec["model_name"]
        else:
            pod_id = deploy_job(group, dry_run_container=args.dry_run_container)
            status = "running"
            for key, spec in group:
                job = cfg[key]
                job["pod_id"] = pod_id
                job["status"] = status
                job["output_model_name"] = spec["model_name"]
        save_yaml(yaml_path, cfg)
        print(f"{label}: wrote status={status} to {yaml_path}")


if __name__ == "__main__":
    main()

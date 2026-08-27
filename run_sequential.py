#!/usr/bin/env python3
"""Sequential cube eval from tester.yaml checkpoints.

    python run_sequential.py
    python run_sequential.py mpc 123 500
    python run_sequential.py plan 123 500 tester.yaml
    python run_sequential.py mpc 42 50 tester.yaml --batch-size 8
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EPOCH_RE = re.compile(r"^weights_epoch_(\d+)\.pt$")
DATASET = "galilai-group/ogb_cube_single"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="mpc", choices=("mpc", "plan"))
    parser.add_argument("seed", nargs="?", type=int, default=42)
    parser.add_argument("num_eval", nargs="?", type=int, default=50)
    parser.add_argument("yaml", nargs="?", default=str(HERE / "tester.yaml"))
    parser.add_argument("--batch-size", type=int, default=50)
    return parser.parse_args()


def load_yaml(path):
    cfg = yaml.safe_load(Path(path).read_text())
    names = [a["output_model_name"] for a in cfg["ablations"]]
    return cfg, names


def npz_path_for(mode, eval_name, output_dir):
    root = Path(output_dir)
    if not root.is_absolute():
        root = HERE / root
    if mode == "plan":
        return root / f"plan_{eval_name}.npz"
    return root / f"{eval_name}.npz"


def highest_local_pt(dest):
    best, best_n = None, -1
    if not dest.is_dir():
        return None
    for path in dest.glob("weights_epoch_*.pt"):
        match = EPOCH_RE.match(path.name)
        if match and int(match.group(1)) > best_n:
            best_n = int(match.group(1))
            best = path
    return best


def ensure_checkpoint(repo, subdir, name, ckpt_root):
    dest = ckpt_root / subdir / name
    pt = highest_local_pt(dest)
    if pt is not None and (dest / "config.json").is_file():
        return pt

    from huggingface_hub import HfApi, hf_hub_download

    prefix = f"{subdir}/{name}/"
    files = HfApi().list_repo_files(repo_id=repo, repo_type="model")
    names = [f[len(prefix):] for f in files if f.startswith(prefix)]
    if "config.json" not in names:
        raise SystemExit(f"no config.json in {repo}/{prefix}")
    epochs = [
        int(match.group(1))
        for n in names
        if (match := EPOCH_RE.match(n))
    ]
    if not epochs:
        raise SystemExit(f"no weights_epoch_*.pt in {repo}/{prefix}")

    epoch = max(epochs)
    for filename in ("config.json", f"weights_epoch_{epoch}.pt"):
        hf_hub_download(
            repo_id=repo,
            filename=f"{prefix}{filename}",
            repo_type="model",
            local_dir=str(ckpt_root),
        )
    return dest / f"weights_epoch_{epoch}.pt"


def run_eval(mode, policy, eval_name, seed, num_eval, batch_size, output_dir):
    if mode == "plan":
        cmd = [
            sys.executable,
            str(HERE / "scripts/plan/eval_wm_cube_plan.py"),
            f"policy={policy}",
            f"eval.dataset_name={DATASET}",
            f"seed={seed}",
            f"eval.name={eval_name}",
            f"eval.num_eval={num_eval}",
            f"eval.batch_size={batch_size}",
            f"eval.output_dir={output_dir}",
            "eval.num_candidates=64",
            "-cn",
            "cube",
        ]
    else:
        cmd = [
            sys.executable,
            str(HERE / "scripts/plan/eval_wm_cube_mpc.py"),
            f"policy={policy}",
            f"eval.dataset_name={DATASET}",
            f"seed={seed}",
            f"eval.name={eval_name}",
            f"eval.num_eval={num_eval}",
            f"eval.batch_size={batch_size}",
            f"eval.output_dir={output_dir}",
            "solver=icem",
            "-cn",
            "cube",
        ]
    subprocess.run(cmd, check=True, cwd=HERE)
    return npz_path_for(mode, eval_name, output_dir)


def hf_eval_dest(subdir, output_dir):
    return f"{subdir}/{output_dir}".strip("/")


def ensure_hf_eval_dir(repo, subdir, output_dir):
    from huggingface_hub import HfApi

    api = HfApi()
    dest_dir = hf_eval_dest(subdir, output_dir)
    api.repo_info(repo_id=repo, repo_type="model")
    existing = api.list_repo_files(repo_id=repo, repo_type="model")
    prefix = dest_dir + "/"
    if any(path.startswith(prefix) for path in existing):
        print(f"HF eval dir exists: {repo}/{dest_dir}")
        return dest_dir
    print(f"creating {repo}/{dest_dir}")
    api.upload_file(
        path_or_fileobj=b"",
        path_in_repo=f"{dest_dir}/.gitkeep",
        repo_id=repo,
        repo_type="model",
    )
    return dest_dir


def push_eval_npzs(repo, subdir, output_dir, npz_paths):
    from huggingface_hub import HfApi

    api = HfApi()
    dest_dir = hf_eval_dest(subdir, output_dir)

    for path in npz_paths:
        path = Path(path)
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        dest = f"{dest_dir}/{path.name}"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=repo,
            repo_type="model",
        )
        print(f"uploaded {dest} to {repo}")


def main():
    args = parse_args()
    yaml_path = Path(args.yaml)
    if not yaml_path.is_file():
        raise SystemExit(f"yaml not found: {yaml_path}")

    cfg, models = load_yaml(yaml_path)
    repo = cfg["hf_repo"]
    subdir = cfg["hf_subdir"]
    output_dir = cfg.get("eval_output_dir", "data")
    eval_output_hf = bool(cfg.get("eval_output_hf", False))
    if "STABLEWM_HOME" not in os.environ:
        raise SystemExit("STABLEWM_HOME is required")
    ckpt_root = Path(os.environ["STABLEWM_HOME"]) / "checkpoints"
    Path(output_dir if Path(output_dir).is_absolute() else HERE / output_dir).mkdir(
        parents=True, exist_ok=True
    )

    print(f"Checking {len(models)} model(s) from {yaml_path}")
    pts = []
    for name in models:
        print(f"  {subdir}/{name}")
        pts.append(ensure_checkpoint(repo, subdir, name, ckpt_root))

    if eval_output_hf:
        ensure_hf_eval_dir(repo, subdir, output_dir)

    suffix = "plan" if args.mode == "plan" else "icem"
    written = []
    try:
        for name, pt in zip(models, pts):
            written.append(
                run_eval(
                    args.mode,
                    pt,
                    f"{name}_{suffix}_{args.seed}",
                    args.seed,
                    args.num_eval,
                    args.batch_size,
                    output_dir,
                )
            )

        written.append(
            run_eval(
                args.mode,
                "quentinll/lewm-cube",
                f"quentinll_{suffix}_cube_{args.seed}",
                args.seed,
                args.num_eval,
                args.batch_size,
                output_dir,
            )
        )
    finally:
        if eval_output_hf and written:
            push_eval_npzs(repo, subdir, output_dir, written)


if __name__ == "__main__":
    main()

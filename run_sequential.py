#!/usr/bin/env python3
"""Sequential cube eval from tester.yaml checkpoints.

    python run_sequential.py
    python run_sequential.py mpc 123 500
    python run_sequential.py plan 123 500 tester.yaml
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
    return parser.parse_args()


def load_yaml(path):
    cfg = yaml.safe_load(Path(path).read_text())
    names = [a["output_model_name"] for a in cfg["ablations"]]
    return cfg["hf_repo"], cfg["hf_subdir"], names


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


def run_eval(mode, policy, eval_name, seed, num_eval):
    if mode == "plan":
        cmd = [
            sys.executable,
            str(HERE / "scripts/plan/eval_wm_cube_plan.py"),
            f"policy={policy}",
            f"eval.dataset_name={DATASET}",
            f"seed={seed}",
            f"eval.name={eval_name}",
            f"eval.num_eval={num_eval}",
            "eval.batch_size=50",
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
            "solver=icem",
            "-cn",
            "cube",
        ]
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    args = parse_args()
    yaml_path = Path(args.yaml)
    if not yaml_path.is_file():
        raise SystemExit(f"yaml not found: {yaml_path}")

    repo, subdir, models = load_yaml(yaml_path)
    ckpt_root = Path(os.environ["STABLEWM_HOME"]) / "checkpoints"

    print(f"Checking {len(models)} model(s) from {yaml_path}")
    pts = []
    for name in models:
        print(f"  {subdir}/{name}")
        pts.append(ensure_checkpoint(repo, subdir, name, ckpt_root))

    suffix = "plan" if args.mode == "plan" else "icem"
    for name, pt in zip(models, pts):
        run_eval(args.mode, pt, f"{name}_{suffix}_{args.seed}", args.seed, args.num_eval)

    run_eval(
        args.mode,
        "quentinll/lewm-cube",
        f"quentinll_{suffix}_cube_{args.seed}",
        args.seed,
        args.num_eval,
    )


if __name__ == "__main__":
    main()

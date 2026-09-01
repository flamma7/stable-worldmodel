#!/usr/bin/env python3
"""Evaluate one HF checkpoint with cube mpc or plan.

Invoked by controller.py, e.g.

    python run_sequential.py mpc visreg_a1.0_lr5e-5_lam0.1 42 50 \\
        --batch-size 50 --hf-repo flamma77/lewm-base --hf-subdir visreg \\
        --eval-output-dir data --dataset galilai-group/ogb_cube_single
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPOCH_RE = re.compile(r"^weights_epoch_(\d+)\.pt$")
DEFAULT_DATASET = "galilai-group/ogb_cube_single"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("mpc", "plan"))
    parser.add_argument("model_name", help="HF folder / output_model_name to download")
    parser.add_argument("seed", nargs="?", type=int, default=42)
    parser.add_argument("num_eval", nargs="?", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--hf-repo", default="flamma77/lewm-base")
    parser.add_argument("--hf-subdir", required=True)
    parser.add_argument("--eval-output-dir", default="data")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=64,
        help="Plan-mode eval.num_candidates (ignored for mpc)",
    )
    parser.add_argument(
        "--is-hf-model",
        action="store_true",
        help="Treat model_name as a HuggingFace repo id (skip checkpoint download)",
    )
    return parser.parse_args()


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


def run_eval(
    mode, policy, eval_name, seed, num_eval, batch_size, output_dir, dataset,
    num_candidates=64,
):
    if mode == "plan":
        cmd = [
            sys.executable,
            str(HERE / "scripts/plan/eval_wm_cube_plan.py"),
            f"policy={policy}",
            f"eval.dataset_name={dataset}",
            f"seed={seed}",
            f"eval.name={eval_name}",
            f"eval.num_eval={num_eval}",
            f"eval.batch_size={batch_size}",
            f"eval.output_dir={output_dir}",
            f"eval.num_candidates={num_candidates}",
            "-cn",
            "cube",
        ]
    else:
        cmd = [
            sys.executable,
            str(HERE / "scripts/plan/eval_wm_cube_mpc.py"),
            f"policy={policy}",
            f"eval.dataset_name={dataset}",
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
    repo = args.hf_repo
    subdir = args.hf_subdir
    output_dir = args.eval_output_dir
    name = args.model_name
    if "STABLEWM_HOME" not in os.environ:
        raise SystemExit("STABLEWM_HOME is required")
    ckpt_root = Path(os.environ["STABLEWM_HOME"]) / "checkpoints"
    Path(output_dir if Path(output_dir).is_absolute() else HERE / output_dir).mkdir(
        parents=True, exist_ok=True
    )

    if args.is_hf_model:
        print(f"Using HuggingFace policy {name}")
        policy = name
    else:
        print(f"Downloading {repo}/{subdir}/{name}")
        policy = ensure_checkpoint(repo, subdir, name, ckpt_root)

    ensure_hf_eval_dir(repo, subdir, output_dir)

    suffix = "plan" if args.mode == "plan" else "icem"
    eval_name = f"{name.replace('/', '-')}_{suffix}_{args.seed}"
    npz = run_eval(
        args.mode,
        policy,
        eval_name,
        args.seed,
        args.num_eval,
        args.batch_size,
        output_dir,
        args.dataset,
        num_candidates=args.num_candidates,
    )
    push_eval_npzs(repo, subdir, output_dir, [npz])


if __name__ == "__main__":
    main()

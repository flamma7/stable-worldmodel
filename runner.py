#!/usr/bin/env python3

"""
Runpod Community Cloud Pod Launcher

Launches a Runpod Community Cloud pod for every ablation in tester.yaml.
Defaults to template my_template and injects LEWM_LR / LEMW_ALPHA /
LEWM_LAMBDA / OUTPUT_MODEL_NAME from the ablation. Pods are named
{name_prension}_a{a}_l{lam}_lr{lr} from tester.yaml.
After launch, waits 60s and checks that the pod is still running.

Ablations that already have a running pod ID are skipped. After launch,
writes the pod ID and output_model_name back onto that ablation.

    Defaults:
    Template: my_template
    GPU:      NVIDIA GeForce RTX 5090
    Region:   Canada
    Config:   tester.yaml (next to this script)
    Name:     {name_prension}_a{a}_l{lam}_lr{lr}
    Wait:     60 seconds, then print success or failure

Features:
    - Resolves a Runpod template by name or ID.
    - Launches pods on Runpod Community Cloud.
    - Supports configurable GPU type and count.
    - Restricts deployment to a selected country/region.
    - Inherits environment variables from the template.
    - Injects LEWM_LR / LEMW_ALPHA / LEWM_LAMBDA / OUTPUT_MODEL_NAME.
    - Allows template environment variables to be overridden with --env.
    - Writes the launched pod ID and output_model_name onto each ablation.
    - Skips ablations whose recorded pod ID is still running.
    - Waits after launch and reports success if the pod is still running.
    - Reads the Runpod API key from the RUNPOD_API_KEY environment variable.

Example:
    python runner.py --region canada
    python runner.py --index 1 --env JOB_ID=123
"""

import argparse
import os
import time
from pathlib import Path

import requests
import yaml

API = "https://rest.runpod.io/v1"
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "tester.yaml"

REGIONS = {
    "canada": "CA",
    "ca": "CA",
    "usa": "US",
    "us": "US",
    "united-states": "US",
    "germany": "DE",
    "de": "DE",
    "france": "FR",
    "fr": "FR",
    "netherlands": "NL",
    "nl": "NL",
    "sweden": "SE",
    "se": "SE",
    "norway": "NO",
    "no": "NO",
}


def api_headers():
    api_key = os.environ["RUNPOD_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def load_config(path):
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "ablations" not in cfg:
        raise RuntimeError(
            f"{path} must be a mapping with an 'ablations' list."
        )

    return cfg


class _IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def save_config(path, cfg):
    path = Path(path)
    with path.open("w") as f:
        yaml.dump(
            cfg,
            f,
            Dumper=_IndentDumper,
            default_flow_style=False,
            sort_keys=False,
        )


def get_ablations(cfg):
    ablations = cfg.get("ablations") or []
    if not isinstance(ablations, list):
        raise RuntimeError("'ablations' must be a list.")
    return ablations


def get_ablation(cfg, index):
    ablations = get_ablations(cfg)

    if index < 0 or index >= len(ablations):
        raise IndexError(
            f"Index {index} is out of range for {len(ablations)} ablations."
        )

    ablation = ablations[index]
    if not isinstance(ablation, dict):
        raise RuntimeError(f"ablations[{index}] must be a mapping.")

    return ablation


def fetch_pod(headers, pod_id):
    r = requests.get(
        f"{API}/pods/{pod_id}",
        headers=headers,
    )

    if r.status_code == 404:
        return None, f"pod not found: {pod_id}"

    if not r.ok:
        return None, f"Runpod returned {r.status_code}: {r.text}"

    return r.json(), None


def running_pod_id(headers, ablation):
    """Return the pod ID if it is still running, else None."""
    pod_id = ablation.get("pod_id")
    if not pod_id:
        return None

    pod, error = fetch_pod(headers, pod_id)
    if error:
        if "not found" in error:
            return None
        raise RuntimeError(error)

    status = (pod.get("desiredStatus") or "").upper()
    if status == "RUNNING":
        return pod_id

    return None


def resolve_template(headers, name_or_id):
    r = requests.get(
        f"{API}/templates",
        headers=headers,
    )
    r.raise_for_status()

    templates = r.json()

    matches = [
        t for t in templates
        if t["name"] == name_or_id or t["id"] == name_or_id
    ]

    if not matches:
        raise RuntimeError(f"Template not found: {name_or_id}")

    return matches[0]


def resolve_region(region):
    region = region.lower()

    if region in REGIONS:
        return REGIONS[region]

    if len(region) == 2:
        return region.upper()

    raise ValueError(
        f"Unknown region '{region}'. "
        "Use a supported region name or 2-letter country code."
    )


def format_name_value(value):
    """Compact ablation value for pod names: 1.0, 0.4, 5e-5."""
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-3:
            mantissa, exponent = f"{value:.0e}".split("e")
            return f"{mantissa}e{int(exponent)}"
        if value == int(value):
            return f"{int(value)}.0"
        return f"{value:g}"
    return str(value)


def pod_name_for(ablation, prefix="start2"):
    a = format_name_value(ablation["a"])
    lam = format_name_value(ablation["lam"])
    lr = format_name_value(ablation["lr"])
    return f"{prefix}_a{a}_l{lam}_lr{lr}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Launch only this ablation index (default: all)",
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to tester.yaml (default: {DEFAULT_CONFIG})",
    )

    parser.add_argument(
        "--template",
        default="my_template",
        help="Runpod template name or ID (default: my_template)",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Pod name prefix (default: name_prension from tester.yaml)",
    )

    parser.add_argument(
        "--gpu",
        default="NVIDIA GeForce RTX 5090",
        help="GPU type (default: RTX 5090)",
    )

    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--region",
        default="canada",
        help="Region/country name or ISO code (default: canada)",
    )

    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable, e.g. --env FOO=bar",
    )

    parser.add_argument(
        "--wait",
        type=int,
        default=60,
        help="Seconds to wait after launch before checking status (default: 60)",
    )

    args = parser.parse_args()

    extra_env = {}
    for item in args.env:
        if "=" not in item:
            raise ValueError(
                f"Invalid --env value '{item}'. Expected KEY=VALUE."
            )
        key, value = item.split("=", 1)
        extra_env[key] = value

    cfg = load_config(args.config)
    ablations = get_ablations(cfg)
    name_prefix = args.name or cfg.get("name_prension")
    if not name_prefix:
        raise RuntimeError(
            f"{args.config} must set name_prension, or pass --name."
        )

    if args.index is not None:
        indices = [args.index]
    else:
        indices = list(range(len(ablations)))

    headers = api_headers()
    template = resolve_template(headers, args.template)
    country_code = resolve_region(args.region)

    launched = 0
    skipped = 0
    successes = 0
    failures = 0

    for index in indices:
        ablation = get_ablation(cfg, index)
        existing = running_pod_id(headers, ablation)
        if existing:
            print(
                f"index {index}: skip, already running ({existing}) "
                f"a={ablation.get('a')} lr={ablation.get('lr')} "
                f"lam={ablation.get('lam')}"
            )
            skipped += 1
            continue

        pod_name = pod_name_for(ablation, prefix=name_prefix)

        env = dict(template.get("env") or {})
        env["LEWM_LR"] = str(ablation["lr"])
        env["LEMW_ALPHA"] = str(ablation["a"])
        env["LEWM_LAMBDA"] = str(ablation["lam"])
        env["OUTPUT_MODEL_NAME"] = pod_name
        env.update(extra_env)

        payload = {
            "name": pod_name,
            "templateId": template["id"],
            "cloudType": "COMMUNITY",
            "gpuTypeIds": [args.gpu],
            "gpuCount": args.gpu_count,
            "countryCodes": [country_code],
            "env": env,
        }

        r = requests.post(
            f"{API}/pods",
            headers=headers,
            json=payload,
        )

        if not r.ok:
            raise RuntimeError(
                f"index {index}: Runpod returned {r.status_code}:\n{r.text}"
            )

        pod = r.json()
        pod_id = pod["id"]
        ablation["pod_id"] = pod_id
        ablation["output_model_name"] = pod_name
        save_config(args.config, cfg)
        launched += 1

        print(f"index {index}: launched {pod_id}")
        print(f"  Ablation: a={ablation['a']} lr={ablation['lr']} lam={ablation['lam']}")
        print(f"  Name:     {pod_name}")
        print(f"  OUTPUT_MODEL_NAME={pod_name}")
        print(f"  Template: {template['name']} ({template['id']})")
        print(f"  GPU:      {args.gpu} x{args.gpu_count}")
        print(f"  Region:   {args.region} ({country_code})")
        print(f"  Cost/hr:  ${pod.get('costPerHr', 'unknown')}")
        print(f"  Waiting {args.wait}s to confirm the pod is still running...")

        time.sleep(args.wait)

        if running_pod_id(headers, ablation):
            print("success")
            successes += 1
        else:
            checked, error = fetch_pod(headers, pod_id)
            status = error or (checked or {}).get("desiredStatus") or "unknown"
            print("failure")
            print(f"  Status:   {status}")
            failures += 1

    print(
        f"Done. launched={launched} skipped={skipped} "
        f"success={successes} failure={failures}"
    )


if __name__ == "__main__":
    main()

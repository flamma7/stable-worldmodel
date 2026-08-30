#!/usr/bin/env python3
"""Launch one Runpod Community Cloud pod with MODE / CMD_* / HF_* env vars.

Used by controller.py. Also:

    python deploy.py --name visreg_a1.0 --gpu "NVIDIA GeForce RTX 5090" \\
        --env MODE=train --env CMD_0="python scripts/train/lewm_visreg.py ..."
"""

import argparse
import os
import time

import requests

API = "https://rest.runpod.io/v1"

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
    "portugal": "PT",
    "pt": "PT",
    "taiwan": "TW",
    "tw": "TW",
}


def api_headers():
    api_key = os.environ["RUNPOD_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def fetch_pod(headers, pod_id):
    r = requests.get(f"{API}/pods/{pod_id}", headers=headers)
    if r.status_code == 404:
        return None, f"pod not found: {pod_id}"
    if not r.ok:
        return None, f"Runpod returned {r.status_code}: {r.text}"
    return r.json(), None


def running_pod_id(headers, pod_id):
    if not pod_id:
        return None
    pod, error = fetch_pod(headers, pod_id)
    if error:
        if "not found" in error:
            return None
        raise RuntimeError(error)
    if (pod.get("desiredStatus") or "").upper() == "RUNNING":
        return pod_id
    return None


def resolve_template(headers, name_or_id):
    r = requests.get(f"{API}/templates", headers=headers)
    r.raise_for_status()
    matches = [
        t for t in r.json()
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
    """Compact value for names: 1.0, 0.4, 5e-5."""
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-3:
            mantissa, exponent = f"{value:.0e}".split("e")
            return f"{mantissa}e{int(exponent)}"
        if value == int(value):
            return f"{int(value)}.0"
        return f"{value:g}"
    return str(value)


def launch_direct(
    name,
    gpu,
    extra_env,
    template="my_template",
    gpu_count=1,
    region="us",
    wait=60,
):
    """Launch one Community Cloud pod. Returns the pod ID."""
    headers = api_headers()
    template_obj = resolve_template(headers, template)
    country_code = resolve_region(region)

    env = dict(template_obj.get("env") or {})
    env.update({str(k): str(v) for k, v in extra_env.items()})

    payload = {
        "name": name,
        "templateId": template_obj["id"],
        "cloudType": "COMMUNITY",
        "gpuTypeIds": [gpu],
        "gpuCount": gpu_count,
        "countryCodes": [country_code],
        "env": env,
    }

    r = requests.post(f"{API}/pods", headers=headers, json=payload)
    if not r.ok:
        raise RuntimeError(f"Runpod returned {r.status_code}:\n{r.text}")

    pod = r.json()
    pod_id = pod["id"]
    print(f"launched {pod_id}")
    print(f"  Name:     {name}")
    print(f"  Template: {template_obj['name']} ({template_obj['id']})")
    print(f"  GPU:      {gpu} x{gpu_count}")
    print(f"  Region:   {region} ({country_code})")
    print(f"  Cost/hr:  ${pod.get('costPerHr', 'unknown')}")
    for key in ("MODE", "HF_DATASET", "HF_DATASET_DIR", "CMD_0"):
        if key in env:
            print(f"  {key}={env[key]}")
    print(f"  Waiting {wait}s to confirm the pod is still running...")

    time.sleep(wait)
    if running_pod_id(headers, pod_id):
        print("success")
    else:
        checked, error = fetch_pod(headers, pod_id)
        status = error or (checked or {}).get("desiredStatus") or "unknown"
        print("failure")
        print(f"  Status:   {status}")
        raise RuntimeError(f"pod {pod_id} is not running ({status})")
    return pod_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Pod name")
    parser.add_argument(
        "--gpu",
        default="NVIDIA GeForce RTX 5090",
        help="GPU type (default: RTX 5090)",
    )
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument(
        "--template",
        default="my_template",
        help="Runpod template name or ID",
    )
    parser.add_argument(
        "--region",
        default="us",
        help="Region/country name or ISO code (us, canada, portugal; default: us)",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable, e.g. --env MODE=train",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=60,
        help="Seconds to wait after launch (default: 60)",
    )
    args = parser.parse_args()

    extra_env = {}
    for item in args.env:
        if "=" not in item:
            raise ValueError(f"Invalid --env value '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        extra_env[key] = value

    launch_direct(
        name=args.name,
        gpu=args.gpu,
        extra_env=extra_env,
        template=args.template,
        gpu_count=args.gpu_count,
        region=args.region,
        wait=args.wait,
    )


if __name__ == "__main__":
    main()

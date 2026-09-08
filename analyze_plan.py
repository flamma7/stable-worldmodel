"""Analyze candidate-plan records from ``eval_wm_cube_plan.py``.

Measures whether the world model's latent planning cost ranks action
sequences the same way the simulator does:

    ρ(C_latent, C_cube)
    ρ(C_latent, C_gripper)
    ρ(C_latent, C_cg)

C_cg = C_cube / s_cube + C_gripper / s_gripper, with fixed scales defaulting
to the median expert start→goal displacements in each npz.

Reads ``plan_*.npz`` (old eval) and ``plan_i_*.npz`` (new eval with start
costs). Files named ``*_<seed>.npz`` (e.g. ``plan_model_plan_42.npz``) are
reported per seed. Scale-consistency checks apply within a seed, not across
seeds.

Higher positive Spearman ρ means the model ranks candidate plans more
usefully for MPC. Correlation is computed per scenario (same start/goal,
varying candidates) then averaged; pooled ρ over all pairs is also reported.

python analyze_plan.py data/
python analyze_plan.py data/ogb_cube_plan.npz
"""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# {anything}_{seed}.npz from run_sequential.py
SEED_RE = re.compile(r'_(\d+)$')

REQUIRED_KEYS = (
    'cost_latent',
    'cost_cube',
    'cost_gripper',
    'cube_displacement',
    'arm_displacement',
)


def per_scenario_spearman(latent, physical):
    """Spearman ρ for each scenario. NaN if either series is constant."""
    n = latent.shape[0]
    rhos = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        x = latent[i]
        y = physical[i]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        rho, _ = spearmanr(x, y)
        rhos[i] = rho
    return rhos


def summarize(rhos):
    valid = np.isfinite(rhos)
    n = int(valid.sum())
    if n == 0:
        return np.nan, np.nan, 0, np.nan
    vals = rhos[valid]
    frac_pos = float((vals > 0).mean())
    return float(vals.mean()), float(vals.std()), n, frac_pos


def pooled_spearman(latent, physical):
    x = np.asarray(latent, dtype=np.float64).reshape(-1)
    y = np.asarray(physical, dtype=np.float64).reshape(-1)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    rho, _ = spearmanr(x, y)
    return float(rho)


def rho_stats(latent, physical):
    rhos = per_scenario_spearman(latent, physical)
    mean, std, n_valid, frac_pos = summarize(rhos)
    pooled = pooled_spearman(latent, physical)
    return {
        'mean': mean,
        'std': std,
        'n_valid': n_valid,
        'frac_pos': frac_pos,
        'pooled': pooled,
    }


def collect_npz_paths(path, prefixes=('plan_',)):
    """Collect ``.npz`` files whose names start with one of ``prefixes``.

    ``plan_`` matches both old ``plan_*`` dumps and new ``plan_i_*`` dumps.
    """
    path = Path(path)
    if path.is_file():
        if path.suffix != '.npz':
            raise SystemExit(f'not an .npz file: {path}')
        if prefixes and not path.name.startswith(prefixes):
            raise SystemExit(
                f'{path.name} does not match prefixes {prefixes}'
            )
        return [path]
    if path.is_dir():
        files = sorted(
            f
            for f in path.glob('*.npz')
            if not prefixes or f.name.startswith(prefixes)
        )
        if not files:
            raise SystemExit(f'no matching .npz files in {path}')
        return files
    raise SystemExit(f'path not found: {path}')


def parse_seed(path, loaded=None):
    match = SEED_RE.search(Path(path).stem)
    if match:
        return int(match.group(1))
    if loaded is not None and 'seed' in loaded.files:
        return int(np.asarray(loaded['seed']).reshape(-1)[0])
    return None


def analyze_npz(path, s_cube=None, s_gripper=None):
    loaded = np.load(path)
    missing = [k for k in REQUIRED_KEYS if k not in loaded.files]
    if missing:
        raise ValueError(f'missing keys {missing}')

    cost_latent = np.asarray(loaded['cost_latent'], dtype=np.float64)
    cost_cube = np.asarray(loaded['cost_cube'], dtype=np.float64)
    cost_gripper = np.asarray(loaded['cost_gripper'], dtype=np.float64)
    cube_d = np.asarray(loaded['cube_displacement'], dtype=np.float64)
    arm_d = np.asarray(loaded['arm_displacement'], dtype=np.float64)

    n_scen, n_cand = cost_latent.shape
    cube_scale = float(s_cube) if s_cube is not None else float(np.median(cube_d))
    grip_scale = (
        float(s_gripper) if s_gripper is not None else float(np.median(arm_d))
    )
    cube_scale = max(cube_scale, 1e-6)
    grip_scale = max(grip_scale, 1e-6)
    cost_cg = cost_cube / cube_scale + cost_gripper / grip_scale

    return {
        'file': path.name,
        'seed': parse_seed(path, loaded),
        'n_scen': n_scen,
        'n_cand': n_cand,
        's_cube': cube_scale,
        's_gripper': grip_scale,
        'cube': rho_stats(cost_latent, cost_cube),
        'gripper': rho_stats(cost_latent, cost_gripper),
        'cg': rho_stats(cost_latent, cost_cg),
    }


def _fmt(value, spec='+.3f'):
    if value is None or not np.isfinite(value):
        return 'n/a'
    return format(value, spec)


def _rho_cell(stats):
    if stats['n_valid'] == 0:
        return 'n/a'
    return (
        f'{_fmt(stats["mean"])}±{_fmt(stats["std"], ".3f")} '
        f'({_fmt(stats["pooled"])})'
    )


def _group_key(row):
    seed = row['seed']
    return (seed is None, -1 if seed is None else seed, row['file'])


def _group_label(row):
    seed = row['seed']
    if seed is None:
        return None
    return f'seed={seed}'


def _check_scale_consistency(rows, s_cube, s_gripper):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row['seed'], []).append(row)

    for seed, group in by_seed.items():
        seed_s = 'unparsed' if seed is None else str(seed)
        ref = group[0]
        for row in group[1:]:
            if s_cube is None and row['s_cube'] != ref['s_cube']:
                raise SystemExit(
                    f's_cube mismatch within seed={seed_s}: '
                    f'{ref["file"]} has {ref["s_cube"]:.8f}, '
                    f'{row["file"]} has {row["s_cube"]:.8f}. '
                    'Pass --s-cube to set a shared scale, or check that '
                    'eval configs (dataset, num_eval) match.'
                )
            if s_gripper is None and row['s_gripper'] != ref['s_gripper']:
                raise SystemExit(
                    f's_gripper mismatch within seed={seed_s}: '
                    f'{ref["file"]} has {ref["s_gripper"]:.8f}, '
                    f'{row["file"]} has {row["s_gripper"]:.8f}. '
                    'Pass --s-gripper to set a shared scale, or check that '
                    'eval configs (dataset, num_eval) match.'
                )


def print_table(rows):
    headers = (
        'file',
        'scen',
        'cand',
        's_cube',
        's_grip',
        'ρ_cube mean±std (pooled)',
        'ρ_grip mean±std (pooled)',
        'ρ_cg mean±std (pooled)',
        '%ρ>0 cg',
    )
    rows = sorted(rows, key=_group_key)
    cells = []
    for row in rows:
        cells.append(
            (
                row['file'],
                str(row['n_scen']),
                str(row['n_cand']),
                _fmt(row['s_cube'], '.4f'),
                _fmt(row['s_gripper'], '.4f'),
                _rho_cell(row['cube']),
                _rho_cell(row['gripper']),
                _rho_cell(row['cg']),
                (
                    'n/a'
                    if not np.isfinite(row['cg']['frac_pos'])
                    else f'{100.0 * row["cg"]["frac_pos"]:.0f}%'
                ),
            )
        )

    widths = [
        max(len(headers[i]), max(len(c[i]) for c in cells))
        for i in range(len(headers))
    ]
    aligns = ['<', '>', '>', '>', '>', '<', '<', '<', '>']

    def fmt_row(vals):
        return '  '.join(
            f'{val:{align}{width}}'
            for val, align, width in zip(vals, aligns, widths)
        )

    rule = '-' * (sum(widths) + 2 * (len(widths) - 1))
    unset = object()
    prev_label = unset
    for row, cell in zip(rows, cells):
        label = _group_label(row)
        if label != prev_label:
            if prev_label is not unset:
                print()
            if label is not None:
                print(label)
            print(fmt_row(headers))
            print(rule)
            prev_label = label
        print(fmt_row(cell))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=Path,
        help='directory of .npz files, or a single .npz',
    )
    parser.add_argument(
        '--s-cube',
        type=float,
        default=None,
        help='C_cg cube scale (default: median expert cube displacement)',
    )
    parser.add_argument(
        '--s-gripper',
        type=float,
        default=None,
        help='C_cg gripper scale (default: median expert arm displacement)',
    )
    args = parser.parse_args()

    paths = collect_npz_paths(args.path)
    rows = []
    skipped = []
    for path in paths:
        try:
            row = analyze_npz(path, args.s_cube, args.s_gripper)
        except (ValueError, OSError, KeyError) as exc:
            skipped.append((path, exc))
            continue
        rows.append(row)

    if not rows:
        raise SystemExit('no valid plan-eval .npz files to analyze')

    _check_scale_consistency(rows, args.s_cube, args.s_gripper)

    n_seeds = len({row['seed'] for row in rows if row['seed'] is not None})
    seed_note = f', {n_seeds} seed(s)' if n_seeds else ''
    print(
        f'Spearman ρ(C_latent, C_physical) over {len(rows)} file(s)'
        f'{seed_note}  [per-scenario mean ± std (pooled)]'
    )
    print()
    print_table(rows)
    if skipped:
        print()
        print('skipped:')
        for path, exc in skipped:
            print(f'  {path.name}')
            # print(f'  {path.name}: {exc}')


if __name__ == '__main__':
    main()

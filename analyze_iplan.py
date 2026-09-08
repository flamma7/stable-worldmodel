"""Analyze start→terminal cost *improvement* from ``eval_wm_cube_plan.py``.

Measures whether the world model knows which plans improve the scene:

    ΔC_cg     = C_cg(x_0)     - C_cg(x_H)
    ΔC_latent = C_latent(z_0) - C_latent(z_hat_H)
    ρ(ΔC_latent, ΔC_cg)

C_cg = C_cube / s_cube + C_gripper / s_gripper, with fixed scales defaulting
to the median expert start→goal displacements in each npz (same as
``analyze_plan.py``).

Needs the start-cost keys written by an updated ``eval_wm_cube_plan.py``:
``cost_latent_start``, ``cost_cube_start``, ``cost_gripper_start``.

Per-scenario ρ of the deltas equals per-scenario ρ of the terminal costs
(``analyze_plan.py``), because C(x_0) is constant within a scenario.
Pooled ρ is a different statistic: it compares improvement across scenes.

Only reads ``plan_i_*.npz`` (the dumps with start-cost keys).

python analyze_iplan.py data/
python analyze_iplan.py data/plan_i_ogb_cube_plan.npz
"""

import argparse
from pathlib import Path

import numpy as np

from analyze_plan import (
    _check_scale_consistency,
    _fmt,
    _group_key,
    _group_label,
    _rho_cell,
    collect_npz_paths,
    parse_seed,
    rho_stats,
)

REQUIRED_KEYS = (
    'cost_latent',
    'cost_cube',
    'cost_gripper',
    'cost_latent_start',
    'cost_cube_start',
    'cost_gripper_start',
    'cube_displacement',
    'arm_displacement',
)


def _as_start(values, n_scen):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n_scen:
        raise ValueError(
            f'start cost has length {arr.shape[0]}, expected {n_scen}'
        )
    return arr


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
    latent_start = _as_start(loaded['cost_latent_start'], n_scen)
    cube_start = _as_start(loaded['cost_cube_start'], n_scen)
    grip_start = _as_start(loaded['cost_gripper_start'], n_scen)

    cube_scale = float(s_cube) if s_cube is not None else float(np.median(cube_d))
    grip_scale = (
        float(s_gripper) if s_gripper is not None else float(np.median(arm_d))
    )
    cube_scale = max(cube_scale, 1e-6)
    grip_scale = max(grip_scale, 1e-6)

    d_latent = latent_start[:, None] - cost_latent
    d_cube = cube_start[:, None] - cost_cube
    d_grip = grip_start[:, None] - cost_gripper
    d_cg = d_cube / cube_scale + d_grip / grip_scale

    return {
        'file': path.name,
        'seed': parse_seed(path, loaded),
        'n_scen': n_scen,
        'n_cand': n_cand,
        's_cube': cube_scale,
        's_gripper': grip_scale,
        'cube': rho_stats(d_latent, d_cube),
        'gripper': rho_stats(d_latent, d_grip),
        'cg': rho_stats(d_latent, d_cg),
    }


def print_table(rows):
    headers = (
        'file',
        'scen',
        'cand',
        's_cube',
        's_grip',
        'ρ_Δcube mean±std (pooled)',
        'ρ_Δgrip mean±std (pooled)',
        'ρ_Δcg mean±std (pooled)',
        '%ρ>0 Δcg',
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

    paths = collect_npz_paths(args.path, prefixes=('plan_i_',))
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
        raise SystemExit(
            'no valid plan-eval .npz files to analyze '
            '(need cost_*_start keys from eval_wm_cube_plan.py)'
        )

    _check_scale_consistency(rows, args.s_cube, args.s_gripper)

    n_seeds = len({row['seed'] for row in rows if row['seed'] is not None})
    seed_note = f', {n_seeds} seed(s)' if n_seeds else ''
    print(
        f'Spearman ρ(ΔC_latent, ΔC_physical) over {len(rows)} file(s)'
        f'{seed_note}  [per-scenario mean ± std (pooled)]'
    )
    print()
    print_table(rows)
    if skipped:
        print()
        print('skipped:')
        for path, exc in skipped:
            print(f'  {path.name}: {exc}')


if __name__ == '__main__':
    main()

"""Analyze anytime-success records from cube eval scripts.

python analyze.py data/
python analyze.py data/ogb_cube_table.npz
"""

import argparse
from pathlib import Path

import numpy as np

DISTANCE_EDGES = (0.04, 0.13, 0.2)
DISTANCE_LABELS = ('0-0.04', '0.04-0.13', '0.13-0.2', '0.2+')


def collect_npz_paths(path):
    path = Path(path)
    if path.is_file():
        if path.suffix != '.npz':
            raise SystemExit(f'not an .npz file: {path}')
        return [path]
    if path.is_dir():
        files = sorted(path.glob('*.npz'))
        if not files:
            raise SystemExit(f'no .npz files in {path}')
        return files
    raise SystemExit(f'path not found: {path}')


def success_by_distance_bin(distances, success):
    bin_idx = np.digitize(distances, DISTANCE_EDGES)
    rows = []
    for i, label in enumerate(DISTANCE_LABELS):
        mask = bin_idx == i
        n = int(mask.sum())
        n_ok = int(success[mask].sum()) if n else 0
        rate = 100.0 * n_ok / n if n else float('nan')
        rows.append((label, n, n_ok, rate))
    return rows


def print_success_chart(rows, title):
    width = 24
    print(title)
    for label, n, n_ok, rate in rows:
        if n == 0:
            bar = ''
            rate_s = '  n/a'
        else:
            bar = '█' * int(round(width * rate / 100.0))
            rate_s = f'{rate:5.1f}%'
        print(f'  {label:<12} | {bar:<{width}} {rate_s}  ({n_ok}/{n})')


def rate(success):
    success = np.asarray(success, dtype=bool)
    n = len(success)
    if n == 0:
        return float('nan')
    return 100.0 * float(success.sum()) / n


def print_gripper_cube_table(gripper_success, cube_success):
    gripper_success = np.asarray(gripper_success, dtype=bool)
    cube_success = np.asarray(cube_success, dtype=bool)
    n = len(gripper_success)

    both = int((gripper_success & cube_success).sum())
    gripper_only = int((gripper_success & ~cube_success).sum())
    cube_only = int((~gripper_success & cube_success).sum())
    neither = int((~gripper_success & ~cube_success).sum())

    def cell(label, count):
        pct = 100.0 * count / max(n, 1)
        return f'{label} {count} ({pct:.1f}%)'

    col_w = 28
    row_w = 18
    rule = '-' * (row_w + 2 * col_w)
    print()
    print(f'{"":<{row_w}} {"Cube success":<{col_w}} {"Cube failure":<{col_w}}')
    print(rule)
    print(
        f'{"Gripper success":<{row_w}} '
        f'{cell("both", both):<{col_w}} '
        f'{cell("gripper-only", gripper_only):<{col_w}}'
    )
    print(rule)
    print(
        f'{"Gripper failure":<{row_w}} '
        f'{cell("cube-only", cube_only):<{col_w}} '
        f'{cell("neither", neither):<{col_w}}'
    )


class IncompatibleNpz(ValueError):
    """npz is not a table-eval dump from eval_wm_cube_mpc.py."""


def load_records(path):
    loaded = np.load(path)
    keys = set(loaded.files)
    if 'cost_latent' in keys and 'records' not in keys:
        raise IncompatibleNpz(
            'plan-eval npz from eval_wm_cube_plan.py (no anytime-success records)'
        )
    if 'records' not in keys:
        raise IncompatibleNpz(
            f'not a table-eval npz (keys: {sorted(keys)})'
        )
    records = loaded['records']
    names = set(records.dtype.names or ())
    if 'cube_displacement' not in names:
        raise IncompatibleNpz("records missing 'cube_displacement'")

    if 'cube_success' in names:
        cube_success = np.asarray(records['cube_success'], dtype=bool)
        gripper_success = np.asarray(records['gripper_success'], dtype=bool)
        both_success = np.asarray(records['both_success'], dtype=bool)
    elif 'success' in names:
        cube_success = np.asarray(records['success'], dtype=bool)
        gripper_success = None
        both_success = None
    else:
        raise IncompatibleNpz("records missing 'cube_success' or 'success'")

    cube_d = np.asarray(records['cube_displacement'])
    arm_d = (
        np.asarray(records['arm_displacement'])
        if 'arm_displacement' in names
        else None
    )
    cube_noop = cube_d < DISTANCE_EDGES[0]
    keep = ~cube_noop
    if arm_d is not None:
        keep = keep & ~(arm_d < DISTANCE_EDGES[0])

    return {
        'file': path.name,
        'n': len(records),
        'n_excl': int(keep.sum()),
        'cube_d': cube_d,
        'arm_d': arm_d,
        'cube_success': cube_success,
        'gripper_success': gripper_success,
        'both_success': both_success,
        'cube_noop': cube_noop,
        'keep': keep,
    }


def print_file_charts(data, exclude_noop):
    cube_d = data['cube_d']
    arm_d = data['arm_d']
    cube_success = data['cube_success']
    gripper_success = data['gripper_success']
    both_success = data['both_success']
    cube_noop = data['cube_noop']
    keep = data['keep']

    print(f'loaded {data["n"]} scenarios from {data["file"]}')

    if exclude_noop:
        n_drop = data['n'] - data['n_excl']
        print(
            f'excluding {n_drop} no-op scenarios '
            f'(displacement < {DISTANCE_EDGES[0]}); '
            f'{data["n_excl"]} remaining'
        )
        cube_d = cube_d[keep]
        cube_success = cube_success[keep]
        if gripper_success is not None:
            gripper_success = gripper_success[keep]
            both_success = both_success[keep]
        if arm_d is not None:
            arm_d = arm_d[keep]

    print(f'cube displacement mean: {cube_d.mean():.6f}')
    print(f'cube displacement std:  {cube_d.std():.6f}')
    if arm_d is not None:
        print(f'arm displacement mean:  {arm_d.mean():.6f}')
        print(f'arm displacement std:   {arm_d.std():.6f}')

    print()
    print(f'cube anytime success:    {rate(data["cube_success"]):5.1f}%')
    print(
        f'cube anytime success (excluding no-op): '
        f'{rate(data["cube_success"][data["keep"]]):5.1f}%'
    )
    if data['gripper_success'] is not None:
        print(
            f'gripper anytime success: {rate(data["gripper_success"]):5.1f}%'
        )
        print(
            f'gripper anytime success (excluding no-op): '
            f'{rate(data["gripper_success"][data["keep"]]):5.1f}%'
        )
        print(f'both anytime success:    {rate(data["both_success"]):5.1f}%')
        print(
            f'both anytime success (excluding no-op): '
            f'{rate(data["both_success"][data["keep"]]):5.1f}%'
        )
    print()
    print_success_chart(
        success_by_distance_bin(cube_d, cube_success),
        'cube anytime success by cube displacement',
    )
    if gripper_success is not None and arm_d is not None:
        print()
        print_success_chart(
            success_by_distance_bin(arm_d, gripper_success),
            'gripper anytime success by arm displacement',
        )
        print()
        print_success_chart(
            success_by_distance_bin(cube_d, both_success),
            'both anytime success by cube displacement',
        )
        print()
        print_success_chart(
            success_by_distance_bin(arm_d, both_success),
            'both anytime success by arm displacement',
        )

    if gripper_success is not None:
        print_gripper_cube_table(gripper_success, cube_success)


def _fmt_rate(value):
    if value is None or not np.isfinite(value):
        return 'n/a'
    return f'{value:5.1f}%'


def print_comparison_table(rows):
    headers = (
        'file',
        'n',
        'n ex-noop',
        'cube',
        'cube ex-noop',
        'grip',
        'grip ex-noop',
        'both',
        'both ex-noop',
    )
    cells = []
    for row in rows:
        cells.append(
            (
                row['file'],
                str(row['n']),
                str(row['n_excl']),
                _fmt_rate(row['cube']),
                _fmt_rate(row['cube_excl']),
                _fmt_rate(row['grip']),
                _fmt_rate(row['grip_excl']),
                _fmt_rate(row['both']),
                _fmt_rate(row['both_excl']),
            )
        )

    widths = [
        max(len(headers[i]), max(len(c[i]) for c in cells))
        for i in range(len(headers))
    ]
    aligns = ['<', '>', '>', '>', '>', '>', '>', '>', '>']

    def fmt_row(vals):
        return '  '.join(
            f'{val:{align}{width}}'
            for val, align, width in zip(vals, aligns, widths)
        )

    rule = '-' * (sum(widths) + 2 * (len(widths) - 1))
    print(fmt_row(headers))
    print(rule)
    for cell in cells:
        print(fmt_row(cell))


def metrics_row(data):
    keep = data['keep']
    grip = data['gripper_success']
    both = data['both_success']
    return {
        'file': data['file'],
        'n': data['n'],
        'n_excl': data['n_excl'],
        'cube': rate(data['cube_success']),
        'cube_excl': rate(data['cube_success'][keep]),
        'grip': rate(grip) if grip is not None else float('nan'),
        'grip_excl': rate(grip[keep]) if grip is not None else float('nan'),
        'both': rate(both) if both is not None else float('nan'),
        'both_excl': rate(both[keep]) if both is not None else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=Path,
        help='directory of .npz files, or a single .npz',
    )
    parser.add_argument(
        '--exclude-noop',
        action='store_true',
        help=(
            'exclude no-op scenarios (cube or arm displacement < '
            f'{DISTANCE_EDGES[0]}) from per-file charts'
        ),
    )
    args = parser.parse_args()

    paths = collect_npz_paths(args.path)
    rows = []
    skipped = []
    for path in paths:
        try:
            data = load_records(path)
        except IncompatibleNpz as exc:
            skipped.append((path, exc))
            continue
        except (ValueError, OSError, KeyError) as exc:
            skipped.append((path, exc))
            continue
        if rows:
            print()
        print('=' * 72)
        print(data['file'])
        print('=' * 72)
        print_file_charts(data, args.exclude_noop)
        rows.append(metrics_row(data))

    if not rows:
        raise SystemExit('no valid eval .npz files to analyze')

    print()
    print('=' * 72)
    print('anytime success comparison  (ex-noop: displacement < '
          f'{DISTANCE_EDGES[0]})')
    print('=' * 72)
    print_comparison_table(rows)
    if skipped:
        print()
        print('skipped:')
        for path, exc in skipped:
            print(f'  {path.name}: {exc}')


if __name__ == '__main__':
    main()

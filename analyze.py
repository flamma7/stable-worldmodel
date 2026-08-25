import argparse
from pathlib import Path

import numpy as np

DISTANCE_EDGES = (0.04, 0.13, 0.2)
DISTANCE_LABELS = ('0-0.04', '0.04-0.13', '0.13-0.2', '0.2+')


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
    return 100.0 * float(success.sum()) / max(len(success), 1)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', type=Path, help='path to eval records .npz')
    parser.add_argument(
        '--exclude-noop',
        action='store_true',
        help=(
            'exclude no-op scenarios (cube or arm displacement < '
            f'{DISTANCE_EDGES[0]}) from all charts'
        ),
    )
    args = parser.parse_args()

    path = args.npz
    loaded = np.load(path)
    records = loaded['records']
    names = set(records.dtype.names)

    print(f'loaded {len(records)} scenarios from {path}')

    if 'cube_success' in names:
        cube_success = np.asarray(records['cube_success'], dtype=bool)
        gripper_success = np.asarray(records['gripper_success'], dtype=bool)
        both_success = np.asarray(records['both_success'], dtype=bool)
    else:
        cube_success = np.asarray(records['success'], dtype=bool)
        gripper_success = None
        both_success = None

    cube_d = records['cube_displacement']
    arm_d = records['arm_displacement'] if 'arm_displacement' in names else None
    cube_noop = cube_d < DISTANCE_EDGES[0]
    arm_noop = arm_d < DISTANCE_EDGES[0] if arm_d is not None else None
    if args.exclude_noop:
        keep = ~cube_noop
        n_cube = int(cube_noop.sum())
        n_arm = 0
        if arm_noop is not None:
            keep = keep & ~arm_noop
            n_arm = int(arm_noop.sum())
        n_drop = int((~keep).sum())
        cube_d = cube_d[keep]
        cube_success = cube_success[keep]
        if gripper_success is not None:
            gripper_success = gripper_success[keep]
            both_success = both_success[keep]
        if arm_d is not None:
            arm_d = arm_d[keep]
        parts = [f'cube displacement < {DISTANCE_EDGES[0]}: {n_cube}']
        if arm_noop is not None:
            parts.append(f'arm displacement < {DISTANCE_EDGES[0]}: {n_arm}')
        print(
            f'excluding {n_drop} no-op scenarios '
            f'({"; ".join(parts)}); '
            f'{len(cube_d)} remaining'
        )

    print(f'cube displacement mean: {cube_d.mean():.6f}')
    print(f'cube displacement std:  {cube_d.std():.6f}')
    if arm_d is not None:
        print(f'arm displacement mean:  {arm_d.mean():.6f}')
        print(f'arm displacement std:   {arm_d.std():.6f}')

    print()
    print(f'cube anytime success:    {rate(cube_success):5.1f}%')
    if not args.exclude_noop:
        print(
            f'cube anytime success (excluding no-op): '
            f'{rate(cube_success[~cube_noop]):5.1f}%'
        )
    if gripper_success is not None:
        print(f'gripper anytime success: {rate(gripper_success):5.1f}%')
        print(f'both anytime success:    {rate(both_success):5.1f}%')
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


if __name__ == '__main__':
    main()

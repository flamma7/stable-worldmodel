import os
from pathlib import Path

import numpy as np

NPZ_OFFSET = Path('checkpoints/quentinll/ogb_cube_results.npz')
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


def print_success_chart(rows):
    width = 24
    print('success rate by cube displacement')
    for label, n, n_ok, rate in rows:
        if n == 0:
            bar = ''
            rate_s = '  n/a'
        else:
            bar = '█' * int(round(width * rate / 100.0))
            rate_s = f'{rate:5.1f}%'
        print(f'  {label:<12} | {bar:<{width}} {rate_s}  ({n_ok}/{n})')


def main():
    home = os.environ.get('STABLEWM_HOME')
    if not home:
        raise EnvironmentError('STABLEWM_HOME is not set')

    path = Path(home) / NPZ_OFFSET
    records = np.load(path)['records']
    distances = records['cube_displacement']
    success = np.asarray(records['success'], dtype=bool)

    print(f'loaded {len(distances)} scenarios from {path}')
    print(f'cube displacement mean: {distances.mean():.6f}')
    print(f'cube displacement std:  {distances.std():.6f}')
    print()
    print_success_chart(success_by_distance_bin(distances, success))


if __name__ == '__main__':
    main()

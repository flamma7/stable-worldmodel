"""Analyze latent-triplet dumps from ``eval_straightness.py``.

For each matching ``.npz`` (must contain ``z_t``, ``z_tp1``, ``z_tp2``;
other eval dumps in the same directory are skipped):

    c_t = cos(z_{t+1} - z_t, z_{t+2} - z_{t+1})

Prints a shared-bin histogram with one column per file.

python analyze_straightness.py data/
python analyze_straightness.py data/ogb_cube_straight.npz
"""

import argparse
from pathlib import Path

import numpy as np

REQUIRED_KEYS = ('z_t', 'z_tp1', 'z_tp2')
BAR_BLOCKS = ' ▏▎▍▌▋▊▉█'
BAR_WIDTH = 16


class IncompatibleNpz(ValueError):
    """npz is not a triplet dump from eval_straightness.py."""


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


def cosine_straightness(z_t, z_tp1, z_tp2):
    """c_t = cos(Δz_t, Δz_{t+1}) in [-1, 1]; NaN if either delta is ~0."""
    v1 = z_tp1 - z_t
    v2 = z_tp2 - z_tp1
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    valid = (n1 > 1e-12) & (n2 > 1e-12)
    c = np.full(z_t.shape[0], np.nan, dtype=np.float64)
    c[valid] = np.sum(v1[valid] * v2[valid], axis=-1) / (n1[valid] * n2[valid])
    return np.clip(c, -1.0, 1.0)


def load_triplets(path):
    loaded = np.load(path)
    keys = set(loaded.files)
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise IncompatibleNpz(f'missing keys {missing}')

    z_t = np.asarray(loaded['z_t'], dtype=np.float64)
    z_tp1 = np.asarray(loaded['z_tp1'], dtype=np.float64)
    z_tp2 = np.asarray(loaded['z_tp2'], dtype=np.float64)
    if z_t.shape != z_tp1.shape or z_t.shape != z_tp2.shape:
        raise ValueError(
            f'shape mismatch z_t={z_t.shape} z_tp1={z_tp1.shape} '
            f'z_tp2={z_tp2.shape}'
        )
    if z_t.ndim != 2:
        raise ValueError(f'expected z_* (N, D), got {z_t.shape}')

    c = cosine_straightness(z_t, z_tp1, z_tp2)
    finite = c[np.isfinite(c)]
    return {
        'file': path.name,
        'n': int(z_t.shape[0]),
        'n_valid': int(finite.size),
        'c': c,
        'mean': float(finite.mean()) if finite.size else float('nan'),
        'std': float(finite.std()) if finite.size else float('nan'),
    }


def histogram(c, edges):
    finite = c[np.isfinite(c)]
    if finite.size == 0:
        return np.zeros(len(edges) - 1, dtype=np.int64)
    counts, _ = np.histogram(finite, bins=edges)
    return counts


def _bin_label(lo, hi, last):
    right = ']' if last else ')'
    return f'[{lo:+.2f},{hi:+.2f}{right}'


def _fmt_pct(count, n):
    if n == 0:
        return '  n/a'
    return f'{100.0 * count / n:5.1f}%'


def _bar(frac, width):
    """Horizontal bar for ``frac`` in [0, 1], using eighth-block glyphs."""
    frac = min(1.0, max(0.0, float(frac)))
    units = frac * width
    full = min(width, int(units))
    rem = units - full
    if full >= width:
        return BAR_BLOCKS[-1] * width
    idx = int(round(rem * (len(BAR_BLOCKS) - 1)))
    filled = BAR_BLOCKS[-1] * full
    if idx > 0 and full < width:
        filled += BAR_BLOCKS[idx]
    return filled + '░' * (width - len(filled))


def _hist_cell(count, n, bar_width):
    if n == 0:
        return f'{_bar(0.0, bar_width)}   n/a'
    return f'{_bar(count / n, bar_width)} {_fmt_pct(count, n)}'


def print_histograms(rows, n_bins, bar_width=BAR_WIDTH):
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    labels = [
        _bin_label(edges[i], edges[i + 1], last=(i + 1 == n_bins))
        for i in range(n_bins)
    ]
    counts = [histogram(row['c'], edges) for row in rows]
    names = [row['file'] for row in rows]
    ns = [row['n_valid'] for row in rows]

    headers = ('bin', *names)
    cells = []
    for i, label in enumerate(labels):
        cells.append(
            (
                label,
                *(
                    _hist_cell(cnt[i], n, bar_width)
                    for cnt, n in zip(counts, ns)
                ),
            )
        )
    cells.append(('n', *(str(row['n_valid']) for row in rows)))
    cells.append(
        (
            'mean',
            *(
                'n/a' if not np.isfinite(row['mean']) else f'{row["mean"]:+.3f}'
                for row in rows
            ),
        )
    )
    cells.append(
        (
            'std',
            *(
                'n/a' if not np.isfinite(row['std']) else f'{row["std"]:.3f}'
                for row in rows
            ),
        )
    )

    widths = [
        max(len(headers[i]), max(len(c[i]) for c in cells))
        for i in range(len(headers))
    ]
    aligns = ['<'] + ['<'] * (len(headers) - 1)
    # summary rows stay right-aligned under the bar+pct block
    summary_aligns = ['<'] + ['>'] * (len(headers) - 1)

    def fmt_row(vals, row_aligns):
        return '  '.join(
            f'{val:{align}{width}}'
            for val, align, width in zip(vals, row_aligns, widths)
        )

    rule = '-' * (sum(widths) + 2 * (len(widths) - 1))
    print(fmt_row(headers, aligns))
    print(rule)
    for i, cell in enumerate(cells):
        if i == n_bins:
            print(rule)
            print(fmt_row(cell, summary_aligns))
        elif i > n_bins:
            print(fmt_row(cell, summary_aligns))
        else:
            print(fmt_row(cell, aligns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=Path,
        help='directory of .npz files, or a single .npz',
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=10,
        help='histogram bins over [-1, 1] (default: 10)',
    )
    parser.add_argument(
        '--bar-width',
        type=int,
        default=BAR_WIDTH,
        help=f'histogram bar width in characters (default: {BAR_WIDTH})',
    )
    args = parser.parse_args()
    if args.bins < 1:
        raise SystemExit('--bins must be >= 1')
    if args.bar_width < 1:
        raise SystemExit('--bar-width must be >= 1')

    paths = collect_npz_paths(args.path)
    rows = []
    skipped = 0
    for path in paths:
        try:
            rows.append(load_triplets(path))
        except IncompatibleNpz:
            skipped += 1
            continue
        except (ValueError, OSError, KeyError) as exc:
            raise SystemExit(f'{path.name}: {exc}') from exc

    if not rows:
        raise SystemExit(
            'no straightness .npz files '
            '(need keys z_t, z_tp1, z_tp2)'
        )

    print(
        f'c_t = cos(z_t+1 - z_t, z_t+2 - z_t+1)  '
        f'{len(rows)} file(s)'
        + (f', skipped {skipped} other .npz' if skipped else '')
    )
    print()
    print_histograms(rows, args.bins, bar_width=args.bar_width)


if __name__ == '__main__':
    main()

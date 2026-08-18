"""Built-in dataset formats. Importing this package registers each format
whose optional dependencies are available; the rest are silently skipped.

Only folder and lerobot ship with the core install. Lance — the default
dataset format — needs lancedb, from the ``[data]`` extra. HDF5, video, and
blob-v2 lance_video need their backing libraries (h5py, torchcodec/imageio,
pylance); install them together with the umbrella extra ``[format]``.
"""

from __future__ import annotations

import logging as _logging


def _try_import(modname: str) -> None:
    try:
        __import__(f'{__name__}.{modname}')
    except ImportError as exc:
        _logging.getLogger(__name__).debug(
            "format '%s' not registered: %s", modname, exc
        )


# Lance needs lancedb + pyarrow from the ``[data]`` extra. Registered first to
# preserve the original registration order: `detect_format` returns the first
# format whose `detect()` matches, and while the built-in predicates are
# mutually exclusive today, keeping the order stable means making Lance
# optional cannot change which format claims a path.
_try_import('lance')

from . import folder  # noqa: E402, F401
from . import lerobot  # noqa: E402, F401

_try_import('hdf5')
_try_import('video')
# Blob-v2 video format depends on pylance + torchcodec/imageio.
_try_import('lance_video')

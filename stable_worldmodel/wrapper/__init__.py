"""Environment wrappers.

The ``default`` wrappers depend only on gymnasium/numpy and are imported
eagerly — :class:`~stable_worldmodel.world.World` needs ``MegaWrapper`` at
import time.

The ``visual`` wrappers are backed by OpenCV, which ships in the ``[env]``
extra rather than the base install. They are therefore resolved lazily
(PEP 562): importing this package — and so importing ``World`` — keeps
working without cv2, and only reaching for a visual wrapper requires it.
"""

import importlib
from typing import TYPE_CHECKING

from stable_worldmodel.wrapper.default import (
    AddPixelsWrapper,
    EnsureGoalInfoWrapper,
    EnsureImageShape,
    EnsureInfoKeysWrapper,
    EverythingToInfoWrapper,
    MapKeysWrapper,
    MegaWrapper,
    ResizeGoalWrapper,
)


# Names living in `.visual`, which imports cv2 at module scope.
_LAZY_VISUAL = frozenset(
    {
        'BlurWrapper',
        'ChromaKeyWrapper',
        'ColorJitterWrapper',
        'CutoutWrapper',
        'GrayscaleWrapper',
        'MovingPatchWrapper',
        'NoiseWrapper',
        'OcclusionWrapper',
        'RandomConvWrapper',
        'RandomShiftWrapper',
        'ResolutionWrapper',
        'constant',
        'cosine',
        'exponential',
        'linear',
        'sinusoidal',
    }
)

if TYPE_CHECKING:
    from stable_worldmodel.wrapper.visual import (
        BlurWrapper,
        ChromaKeyWrapper,
        ColorJitterWrapper,
        CutoutWrapper,
        GrayscaleWrapper,
        MovingPatchWrapper,
        NoiseWrapper,
        OcclusionWrapper,
        RandomConvWrapper,
        RandomShiftWrapper,
        ResolutionWrapper,
        constant,
        cosine,
        exponential,
        linear,
        sinusoidal,
    )


def __getattr__(name: str):
    if name in _LAZY_VISUAL:
        try:
            mod = importlib.import_module('stable_worldmodel.wrapper.visual')
        except ImportError as exc:
            raise ImportError(
                f'{name!r} is an OpenCV-backed visual wrapper and needs the '
                "'env' extra: pip install 'stable-worldmodel[env]' "
                f'(original error: {exc})'
            ) from exc
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))


__all__ = [
    'AddPixelsWrapper',
    'BlurWrapper',
    'ChromaKeyWrapper',
    'ColorJitterWrapper',
    'CutoutWrapper',
    'EnsureGoalInfoWrapper',
    'EnsureImageShape',
    'EnsureInfoKeysWrapper',
    'EverythingToInfoWrapper',
    'GrayscaleWrapper',
    'MapKeysWrapper',
    'MegaWrapper',
    'MovingPatchWrapper',
    'NoiseWrapper',
    'OcclusionWrapper',
    'RandomConvWrapper',
    'RandomShiftWrapper',
    'ResizeGoalWrapper',
    'ResolutionWrapper',
    'constant',
    'cosine',
    'exponential',
    'linear',
    'sinusoidal',
]

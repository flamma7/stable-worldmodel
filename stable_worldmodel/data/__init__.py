from .utils import *  # noqa: F403
from .dataset import *  # noqa: F403
from .normalization import (
    IdentityScaler,
    PercentileScaler,
    ZScoreScaler,
    get_scaler,
)
from .utils import column_normalizer
from .buffer import ReplayBuffer, classic_filter
from .format import (
    EPISODE_DATA_KEY,
    FORMATS,
    WRITE_MODES,
    Format,
    Writer,
    detect_format,
    get_format,
    list_formats,
    register_format,
    split_episode_data,
    validate_write_mode,
)

# Importing the formats subpackage registers all built-in formats whose
# optional deps are installed.
from . import formats as _formats  # noqa: F401

# Re-export concrete readers/writers from their format modules so existing
# imports like `from stable_worldmodel.data import LanceDataset` keep working.
# Optional formats (lance, hdf5, video) are re-exported only when their extras
# are installed; absent ones are simply not bound at module level.
from .formats.folder import FolderDataset, FolderWriter, ImageDataset
from .formats.lerobot import LeRobotAdapter

try:
    from .formats.lance import LanceDataset, LanceWriter  # noqa: F401
except ImportError:
    pass

try:
    from .formats.hdf5 import HDF5Dataset, HDF5Writer  # noqa: F401
except ImportError:
    pass

try:
    from .formats.video import VideoDataset, VideoWriter  # noqa: F401
except ImportError:
    pass

try:
    from .formats.lance_video import (  # noqa: F401
        LanceVideoDataset,
        LanceVideoWriter,
    )
except ImportError:
    pass


__all__ = [
    'EPISODE_DATA_KEY',
    'FORMATS',
    'Format',
    'FolderDataset',
    'FolderWriter',
    'IdentityScaler',
    'ImageDataset',
    'LeRobotAdapter',
    'PercentileScaler',
    'ReplayBuffer',
    'WRITE_MODES',
    'Writer',
    'ZScoreScaler',
    'classic_filter',
    'column_normalizer',
    'detect_format',
    'get_format',
    'get_scaler',
    'list_formats',
    'register_format',
    'split_episode_data',
    'validate_write_mode',
]


# ``LanceDataset``/``LanceWriter`` are bound above only when the ``[data]``
# extra is installed. Appending them conditionally (rather than listing them
# unconditionally in __all__) keeps ``import *`` from raising AttributeError
# on a base install. hdf5/video/lance_video were never in __all__ and stay out.
if 'LanceDataset' in globals():
    __all__ += ['LanceDataset', 'LanceWriter']

import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

os.environ['MUJOCO_GL'] = 'glfw'

import hydra
import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='reacher')
def run(cfg):
    """Collect random trajectories from the Reacher environment."""

    world = swm.World(cfg.env_name, **cfg.world)

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None

    rng = np.random.default_rng(cfg.seed)

    world.set_policy(RandomPolicy(seed=rng.integers(0, 1_000_000).item()))

    world.collect(
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'dmc/reacher_random.lance',
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    logger.info('Completed random data collection for reacher')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s',
    )
    run()

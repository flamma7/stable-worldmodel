from pathlib import Path

import hydra
import logging
import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    """Run data collection script"""

    world = swm.World('swm/PushT-v1', **cfg.world, render_mode='rgb_array')
    world.set_policy(WeakPolicy(dist_constraint=100))

    rng = np.random.default_rng(cfg.seed)

    for i in range(10):
        world.collect(
            Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
            / 'datasets'
            / f'pusht_toy/shard_{i}.lance',
            episodes=500,
            seed=rng.integers(0, 1_000_000).item(),
        )

    logger.info(' 🎉🎉🎉 Completed data collection for pusht_toy 🎉🎉🎉')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s',
    )
    run()

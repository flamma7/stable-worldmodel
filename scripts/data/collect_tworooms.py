from pathlib import Path

import hydra
import logging
import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.envs.two_room import ExpertPolicy

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    """Run data collection script"""

    world = swm.World('swm/TwoRoom-v1', **cfg.world, render_mode='rgb_array')
    world.set_policy(ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05))

    options = cfg.get('options')
    rng = np.random.default_rng(cfg.seed)

    world.collect(
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'tworoom_expert.lance',
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    logger.info(' 🎉🎉🎉 Completed data collection for tworoom 🎉🎉🎉')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s',
    )
    run()

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
    world.set_policy(ExpertPolicy())

    variation_list = list(world.envs.single_variation_space.names())
    variation_default = {
        'agent.position',
        'target.position',
        'door.size',
        'door.position',
    }

    # exclude default variations
    variation_list = set(variation_list)
    rng = np.random.default_rng(cfg.seed)

    for var in variation_list:
        var = var.replace('variation.', '')
        if var in variation_default:
            continue
        world = swm.World(
            'swm/TwoRoom-v1', **cfg.world, render_mode='rgb_array'
        )
        world.set_policy(ExpertPolicy())
        print(f'Collecting data for variable: {var}')
        var_name = var.replace('.', '/')
        world.collect(
            Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
            / 'datasets'
            / f'tworoom_fov/{var_name}.lance',
            episodes=cfg.num_traj,
            seed=rng.integers(0, 1_000_000).item(),
            options={'variation': tuple([var] + list(variation_default))},
        )

        logger.info(
            f' 🎉🎉🎉 Completed data collection for tworoom {var_name} 🎉🎉🎉'
        )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s',
    )
    run()

"""
python scripts/plan/eval_wm_cube.py policy=quentinll/lewm-cube eval.name=ogb_cube_results eval.dataset_name=galilai-group/ogb_cube_single seed=42 eval.num_eval=50 -cn cube
"""

"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def dataset_columns(dataset):
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return names


def episode_col(dataset):
    names = dataset_columns(dataset)
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def cube_pos_col(dataset):
    names = dataset_columns(dataset)
    for col in (
        'privileged_block_0_pos',
        'privileged/block_0_pos',
    ):
        if col in names:
            return col
    return None


def target_cube_displacement(dataset, start_rows, goal_offset, cube_col):
    """L2 distance the cube travels on the expert trajectory (start → goal)."""
    pos = np.asarray(dataset.get_col_data(cube_col), dtype=np.float64)
    pos = np.reshape(pos, (pos.shape[0], -1))
    goal_rows = start_rows + goal_offset
    step = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    if not np.array_equal(step[goal_rows], step[start_rows] + goal_offset):
        raise ValueError(
            'Goal rows are not start_row + goal_offset; the dataset is '
            'not stored episode-contiguously.'
        )
    start_pos = pos[start_rows]
    goal_pos = pos[goal_rows]
    return np.linalg.norm(goal_pos - start_pos, axis=-1)


def img_transform(cfg, dtype=torch.float32):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = episode_col(dataset)
    # col_name = (
    #     'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    # )

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    return dataset


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = episode_col(dataset)
    # col_name = (
    #     'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    # )
    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name), return_index=True
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # -- run evaluation
    policy = cfg.get('policy', 'random')

    if policy != 'random':
        model = swm.wm.utils.load_pretrained(cfg.policy)
        if cfg.get('bf16', False):
            model = model.to(torch.bfloat16)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get('compile', False):
            encoder_attr = (
                'backbone' if hasattr(model, 'backbone') else 'encoder'
            )
            setattr(
                model,
                encoder_attr,
                torch.compile(getattr(model, encoder_attr)),
            )
            model.predictor = torch.compile(model.predictor)
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = episode_col(dataset)
    # col_name = (
    #     'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    # )
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices), size=cfg.eval.num_eval, replace=False
    )
    # random_episode_indices = g.choice(
    #     len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    # )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_col_data(col_name)[random_episode_indices]
    eval_start_idx = dataset.get_col_data('step_idx')[random_episode_indices]
    # eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    # eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']

    cube_col = cube_pos_col(dataset)
    cube_displacement = None
    if cube_col is not None:
        cube_displacement = target_cube_displacement(
            dataset,
            random_episode_indices,
            cfg.eval.goal_offset_steps,
            cube_col,
        )

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)
    print(
        f'[eval] saving videos to {results_path.resolve()} '
        '(one env_{i}.mp4 per env)'
    )

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    if cfg.get('compile', False):
        print('Warming up compiled model...')
        warmup_autocast_ctx = torch.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=cfg.get('bf16', False),
        )
        with warmup_autocast_ctx:
            n = world.num_envs
            world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist()[:n],
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist()[:n],
                callables=OmegaConf.to_container(
                    cfg.eval.get('callables'), resolve=True
                ),
                video=results_path,
            )
        print('Warmup done.')

    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            ),
            video=results_path,
        )
    end_time = time.time()

    print(metrics)
    print(f'[eval] videos saved to {results_path.resolve()}')

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')

    n = len(eval_episodes)
    successes = np.asarray(metrics['episode_successes']).reshape(n).astype(
        bool
    )
    records = np.empty(
        n,
        dtype=[
            ('scenario', np.int32),
            ('episode_idx', np.int64),
            ('start_step', np.int32),
            ('cube_displacement', np.float32),
            ('success', np.bool_),
        ],
    )
    records['scenario'] = np.arange(n, dtype=np.int32)
    records['episode_idx'] = np.asarray(eval_episodes).reshape(n)
    records['start_step'] = np.asarray(eval_start_idx).reshape(n)
    records['cube_displacement'] = (
        cube_displacement.astype(np.float32)
        if cube_displacement is not None
        else np.full(n, np.nan, dtype=np.float32)
    )
    records['success'] = successes

    npz_path = Path(hydra.utils.get_original_cwd()) / f'{cfg.eval.name}.npz'
    np.savez(npz_path, records=records)
    print(f'[eval] per-scenario records saved to {npz_path.resolve()}')


if __name__ == '__main__':
    run()

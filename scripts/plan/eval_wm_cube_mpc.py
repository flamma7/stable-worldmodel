"""
python scripts/plan/eval_wm_cube_mpc.py policy=quentinll/lewm-cube eval.name=ogb_cube_table eval.dataset_name=galilai-group/ogb_cube_single seed=42 eval.num_eval=50 eval.batch_size=50 -cn cube
"""

"""Evaluate a World Model with one full-budget MPC rollout per scenario.

Each episode is rolled out once, never terminated early. Cube and gripper
are scored on that same trajectory (0.04m anytime threshold). Writes a
``.npz`` with per-scenario records for later analysis:

- cube_success: cube within threshold at any step
- gripper_success: gripper within the same threshold at any step
- both_success: cube_success AND gripper_success
- cube_displacement: expert ||goal_cube - start_cube||
- arm_displacement: expert ||goal_gripper - start_gripper||
"""

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

# Same 0.04m threshold CubeEnv uses for cube-to-target success.
SUCCESS_THRESHOLD = 0.04


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


def gripper_pos_col(dataset):
    names = dataset_columns(dataset)
    for col in (
        'proprio_effector_pos',
        'proprio/effector_pos',
    ):
        if col in names:
            return col
    return None


def _assert_contiguous_goals(dataset, start_rows, goal_offset):
    goal_rows = start_rows + goal_offset
    step = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    if not np.array_equal(step[goal_rows], step[start_rows] + goal_offset):
        raise ValueError(
            'Goal rows are not start_row + goal_offset; the dataset is '
            'not stored episode-contiguously.'
        )
    return goal_rows


def _positions_at(dataset, col, rows):
    pos = np.asarray(dataset.get_col_data(col), dtype=np.float64)
    pos = np.reshape(pos, (pos.shape[0], -1))
    return pos[rows]


def target_displacement(dataset, start_rows, goal_offset, col):
    """L2 distance from start to goal on the expert trajectory."""
    goal_rows = _assert_contiguous_goals(dataset, start_rows, goal_offset)
    start_pos = _positions_at(dataset, col, start_rows)
    goal_pos = _positions_at(dataset, col, goal_rows)
    return np.linalg.norm(goal_pos - start_pos, axis=-1)


def live_cube_positions(world):
    """Cube-0 XYZ of each env at the current sim state."""
    pos = np.empty((world.num_envs, 3), dtype=np.float64)
    for i, env in enumerate(world.envs.envs):
        unwrapped = env.unwrapped
        pos[i] = unwrapped._data.joint('object_joint_0').qpos[:3]
    return pos


def live_gripper_positions(world):
    """Pinch-site XYZ of each env at the current sim state."""
    pos = np.empty((world.num_envs, 3), dtype=np.float64)
    for i, env in enumerate(world.envs.envs):
        unwrapped = env.unwrapped
        pos[i] = unwrapped._data.site_xpos[unwrapped._pinch_site_id]
    return pos


def build_world(cfg, num_envs):
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg['num_envs'] = num_envs
    # Full eval budget so cube and gripper are scored on the same unfrozen
    # trajectory (cube success must not freeze the remaining steps).
    world_cfg['terminate_at_goal'] = False
    return swm.World(**world_cfg, image_shape=(224, 224))


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

    # create world environment (capped so num_eval does not spawn that many envs)
    batch_size = min(
        int(cfg.eval.get('batch_size', cfg.eval.num_eval)),
        int(cfg.eval.num_eval),
    )
    world = build_world(cfg, batch_size)
    print(
        f'[eval] {cfg.eval.num_eval} scenarios, '
        f'{batch_size} parallel envs '
        f'({(int(cfg.eval.num_eval) + batch_size - 1) // batch_size} batches)'
    )

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
        drop = (
            ('motion_encoder',)
            if cfg.get('drop_motion_encoder', True)
            else None
        )
        model = swm.wm.utils.load_pretrained(cfg.policy, drop_modules=drop)
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
    if cube_col is None:
        raise ValueError(
            'Dataset has no cube position column '
            '(expected privileged_block_0_pos or privileged/block_0_pos).'
        )
    gripper_col = gripper_pos_col(dataset)
    if gripper_col is None:
        raise ValueError(
            'Dataset has no gripper effector position column '
            '(expected proprio_effector_pos or proprio/effector_pos).'
        )

    goal_offset = cfg.eval.goal_offset_steps
    cube_displacement = target_displacement(
        dataset, random_episode_indices, goal_offset, cube_col
    )
    arm_displacement = target_displacement(
        dataset, random_episode_indices, goal_offset, gripper_col
    )
    goal_rows = _assert_contiguous_goals(
        dataset, random_episode_indices, goal_offset
    )
    goal_cube = _positions_at(dataset, cube_col, goal_rows)
    goal_gripper = _positions_at(dataset, gripper_col, goal_rows)

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)
    print(
        f'[eval] saving videos to {results_path.resolve()} '
        '(one env_{i}.mp4 per env, under batch_* dirs)'
    )

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    eval_episodes_list = eval_episodes.tolist()
    eval_start_list = eval_start_idx.tolist()
    n_eval = len(eval_episodes_list)
    callables = OmegaConf.to_container(
        cfg.eval.get('callables'), resolve=True
    )
    n_batches = (n_eval + batch_size - 1) // batch_size

    def run_chunk(eval_world, start, end, video_dir, track=True):
        n = end - start
        cube_any = np.zeros(n, dtype=bool)
        grip_any = np.zeros(n, dtype=bool)
        orig_run = eval_world._run

        def tracked_run(*args, **kwargs):
            goal_c = goal_cube[start:end]
            goal_g = goal_gripper[start:end]

            def mark(world):
                cube_err = np.linalg.norm(
                    live_cube_positions(world) - goal_c, axis=-1
                )
                grip_err = np.linalg.norm(
                    live_gripper_positions(world) - goal_g, axis=-1
                )
                cube_any[:] |= cube_err <= SUCCESS_THRESHOLD
                grip_any[:] |= grip_err <= SUCCESS_THRESHOLD

            mark(eval_world)
            orig_on_step = kwargs.get('on_step')

            def on_step(world):
                mark(world)
                if orig_on_step is not None:
                    orig_on_step(world)

            kwargs['on_step'] = on_step
            return orig_run(*args, **kwargs)

        if track:
            eval_world._run = tracked_run
        try:
            metrics = eval_world.evaluate(
                dataset=dataset,
                start_steps=eval_start_list[start:end],
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes_list[start:end],
                callables=callables,
                video=video_dir,
            )
        finally:
            eval_world._run = orig_run
        return metrics, cube_any, grip_any

    if cfg.get('compile', False):
        print('Warming up compiled model...')
        warmup_autocast_ctx = torch.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=cfg.get('bf16', False),
        )
        with warmup_autocast_ctx:
            run_chunk(world, 0, world.num_envs, None, track=False)
        print('Warmup done.')

    start_time = time.time()
    metric_chunks = []
    cube_chunks = []
    grip_chunks = []
    eval_world = world
    with autocast_ctx:
        for batch_idx, start in enumerate(range(0, n_eval, batch_size)):
            end = min(start + batch_size, n_eval)
            chunk_n = end - start
            if chunk_n != eval_world.num_envs:
                eval_world.close()
                eval_world = build_world(cfg, chunk_n)
                eval_world.set_policy(policy)
            print(
                f'[eval] batch {batch_idx + 1}/{n_batches} '
                f'({start}:{end} of {n_eval})'
            )
            metrics, cube_any, grip_any = run_chunk(
                eval_world,
                start,
                end,
                results_path / f'batch_{start:04d}',
            )
            metric_chunks.append(metrics)
            cube_chunks.append(cube_any)
            grip_chunks.append(grip_any)
    end_time = time.time()

    cube_success = np.concatenate(cube_chunks)
    gripper_success = np.concatenate(grip_chunks)
    both_success = cube_success & gripper_success
    metrics = {
        'cube_success': cube_success,
        'gripper_success': gripper_success,
        'both_success': both_success,
        'cube_success_rate': float(cube_success.sum()) / n_eval * 100.0,
        'gripper_success_rate': float(gripper_success.sum()) / n_eval * 100.0,
        'both_success_rate': float(both_success.sum()) / n_eval * 100.0,
        'success_threshold': SUCCESS_THRESHOLD,
    }
    seeds = [m.get('seeds') for m in metric_chunks]
    if all(s is not None for s in seeds):
        metrics['seeds'] = np.concatenate(
            [np.asarray(s).reshape(-1) for s in seeds]
        )

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
    records = np.empty(
        n,
        dtype=[
            ('scenario', np.int32),
            ('episode_idx', np.int64),
            ('start_step', np.int32),
            ('cube_displacement', np.float32),
            ('arm_displacement', np.float32),
            ('cube_success', np.bool_),
            ('gripper_success', np.bool_),
            ('both_success', np.bool_),
        ],
    )
    records['scenario'] = np.arange(n, dtype=np.int32)
    records['episode_idx'] = np.asarray(eval_episodes).reshape(n)
    records['start_step'] = np.asarray(eval_start_idx).reshape(n)
    records['cube_displacement'] = cube_displacement.astype(np.float32)
    records['arm_displacement'] = arm_displacement.astype(np.float32)
    records['cube_success'] = cube_success.reshape(n)
    records['gripper_success'] = gripper_success.reshape(n)
    records['both_success'] = both_success.reshape(n)

    output_dir = getattr(cfg.eval, 'output_dir', 'data')
    npz_path = (
        Path(hydra.utils.get_original_cwd()) / output_dir / f'{cfg.eval.name}.npz'
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        records=records,
        success_threshold=np.float32(SUCCESS_THRESHOLD),
    )
    print(f'[eval] per-scenario records saved to {npz_path.resolve()}')


if __name__ == '__main__':
    run()

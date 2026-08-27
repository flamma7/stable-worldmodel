"""
python scripts/plan/eval_wm_cube_plan.py policy=quentinll/lewm-cube eval.name=ogb_cube_plan eval.dataset_name=galilai-group/ogb_cube_single seed=42 eval.num_eval=50 eval.batch_size=50 eval.num_candidates=64 -cn cube
"""

"""Collect candidate-plan costs for a world model.

``eval.num_eval`` is the number of start/goal scenarios.
``eval.num_candidates`` is the number of random action sequences sampled
per scenario (N below). Spearman ρ is computed across those N plans.

For each dataset scenario (fixed start and goal), sample N action sequences,
score them with the world model's latent planning cost, then execute the same
sequences in the simulator and record terminal cube / gripper distances.

Writes ``<eval.output_dir>/plan_<eval.name>.npz`` for ``analyze_plan.py``. Per-scenario arrays:

- cost_latent: (N,)  |z_hat_H - z_g|_2^2  from ``model.get_cost``
- cost_cube: (N,)    |p_cube,H - p_cube,g|_2
- cost_gripper: (N,) |p_gripper,H - p_gripper,g|_2
"""

import gymnasium as gym
gym.logger.set_level(gym.logger.ERROR)
import warnings
warnings.filterwarnings(
    "ignore",
    message="lancedb fork support is experimental.*",
    category=RuntimeWarning,
)

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
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal


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


def build_world(cfg, num_envs, plan_len):
    cfg.world.max_episode_steps = 2 * plan_len
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg['num_envs'] = num_envs
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


def fit_action_scaler(dataset, keys_to_cache):
    process = {}
    for col in keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != 'action':
            process[f'goal_{col}'] = process[col]
    return process


def sample_eval_rows(dataset, cfg):
    col_name = episode_col(dataset)
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    chosen = g.choice(len(valid_indices), size=cfg.eval.num_eval, replace=False)
    rows = np.sort(valid_indices[chosen])
    episodes = dataset.get_col_data(col_name)[rows]
    start_idx = dataset.get_col_data('step_idx')[rows]
    if len(episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )
    return rows, episodes, start_idx


def setup_from_dataset(world, init_state, goal_state, callables):
    """Reset envs to dataset start/goal (same path as World.evaluate)."""
    world.reset(seed=init_state.get('seed'))
    n = world.num_envs
    if callables:
        merged = {**init_state, **goal_state}
        for i in range(n):
            env_init = {k: v[i] for k, v in merged.items()}
            _apply_callables(world.envs.envs[i].unwrapped, callables, env_init)
    shape_prefix = world.infos['pixels'].shape[:2]
    for src in (init_state, goal_state):
        for k, v in src.items():
            if k in world.infos or k in goal_state:
                world.infos[k] = np.broadcast_to(
                    v[:, None, ...], shape_prefix + v.shape[1:]
                ).copy()


def slice_info(info, i):
    out = {}
    for k, v in info.items():
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            out[k] = v[i : i + 1]
        elif isinstance(v, list):
            out[k] = [v[i]]
        else:
            out[k] = v
    return out


def expand_samples(info, n_samples, device, dtype):
    """Unsqueeze a sample dim and expand, matching CEMSolver."""
    out = {}
    for k, v in info.items():
        if torch.is_tensor(v):
            target_dtype = dtype if v.is_floating_point() else None
            v = v.to(device=device, dtype=target_dtype)
            v = v.unsqueeze(1).expand(v.shape[0], n_samples, *v.shape[1:])
        elif isinstance(v, np.ndarray):
            v = np.repeat(v[:, None, ...], n_samples, axis=1)
        out[k] = v
    return out


def latent_costs(model, info, candidates, device, dtype):
    """World-model planning cost for each (scenario, candidate).

    Processes one scenario at a time so GPU memory matches CEM's default
    ``batch_size=1`` (N candidates, not B*N).
    """
    bsz, n_cand = candidates.shape[:2]
    costs = np.empty((bsz, n_cand), dtype=np.float32)
    for i in range(bsz):
        expanded = expand_samples(slice_info(info, i), n_cand, device, dtype)
        expanded.pop('emb', None)
        expanded.pop('goal_emb', None)
        expanded.pop('predicted_emb', None)
        cost = model.get_cost(expanded, candidates[i : i + 1])
        costs[i] = cost.detach().float().cpu().numpy().reshape(n_cand)
    return costs


def blocked_to_env_actions(candidates, process, action_dim, action_block, space):
    """Inverse-transform blocked WM actions to clipped env-step actions."""
    x = candidates.detach().float().cpu().numpy()
    bsz, n_cand, horizon, _ = x.shape
    plan_len = horizon * action_block
    x = x.reshape(bsz, n_cand, plan_len, action_dim)
    if 'action' in process:
        flat = x.reshape(-1, action_dim)
        flat = process['action'].inverse_transform(flat)
        x = flat.reshape(bsz, n_cand, plan_len, action_dim)
    x = np.clip(x, space.low, space.high)
    return x.astype(np.float32, copy=False)


def execute_plan(world, actions):
    """Open-loop env steps. ``actions`` is (num_envs, plan_len, action_dim)."""
    for t in range(actions.shape[1]):
        _, _, _, _, infos = world.envs.step(actions[:, t])
        world.infos = infos


@hydra.main(version_base=None, config_path='./config', config_name='cube')
def run(cfg: DictConfig):
    policy_name = cfg.get('policy', 'random')
    if policy_name == 'random':
        raise ValueError(
            'eval_wm_cube_plan.py requires a world-model checkpoint '
            '(policy=<run-or-hf-id>), not policy=random.'
        )

    horizon = int(cfg.plan_config.horizon)
    action_block = int(cfg.plan_config.action_block)
    plan_len = horizon * action_block
    n_eval = int(cfg.eval.num_eval)
    batch_size = min(int(cfg.eval.get('batch_size', n_eval)), n_eval)
    solver_cfg = cfg.get('solver')
    n_cand = int(
        cfg.eval.get(
            'num_candidates',
            solver_cfg.get('num_samples', 64) if solver_cfg else 64,
        )
    )
    var_scale = float(
        cfg.eval.get(
            'var_scale',
            solver_cfg.get('var_scale', 1.0) if solver_cfg else 1.0,
        )
    )
    n_batches = (n_eval + batch_size - 1) // batch_size

    world = build_world(cfg, batch_size, plan_len)
    print(
        f'[eval] {n_eval} scenarios, {n_cand} candidates, '
        f'{batch_size} parallel envs, plan_len={plan_len} '
        f'({n_batches} batches)'
    )

    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = fit_action_scaler(dataset, cfg.dataset.keys_to_cache)
    prep = swm.policy.BasePolicy()
    prep.process = process
    prep.transform = transform

    rows, eval_episodes, eval_start_idx = sample_eval_rows(dataset, cfg)
    print(rows)

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
        dataset, rows, goal_offset, cube_col
    )
    arm_displacement = target_displacement(
        dataset, rows, goal_offset, gripper_col
    )
    goal_rows = _assert_contiguous_goals(dataset, rows, goal_offset)
    goal_cube = _positions_at(dataset, cube_col, goal_rows)
    goal_gripper = _positions_at(dataset, gripper_col, goal_rows)

    drop = (
        ('motion_encoder',)
        if cfg.get('drop_motion_encoder', True)
        else None
    )
    model = swm.wm.utils.load_pretrained(cfg.policy, drop_modules=drop)
    if not hasattr(model, 'get_cost'):
        raise TypeError(
            'Loaded checkpoint has no get_cost; need a world model.'
        )
    if cfg.get('bf16', False):
        model = model.to(torch.bfloat16)
    model = model.to('cuda')
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if cfg.get('compile', False):
        encoder_attr = 'backbone' if hasattr(model, 'backbone') else 'encoder'
        setattr(
            model,
            encoder_attr,
            torch.compile(getattr(model, encoder_attr)),
        )
        model.predictor = torch.compile(model.predictor)

    enc_dim = getattr(model.action_encoder, 'input_dim', None)
    action_dim = int(np.prod(world.envs.single_action_space.shape))
    blocked_dim = action_dim * action_block
    if enc_dim is not None and int(enc_dim) != blocked_dim:
        raise ValueError(
            f'action_encoder.input_dim={enc_dim} does not match '
            f'action_dim*action_block={action_dim}*{action_block}={blocked_dim}.'
        )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed))
    callables = OmegaConf.to_container(cfg.eval.get('callables'), resolve=True)
    eval_episodes_list = np.asarray(eval_episodes).tolist()
    eval_start_list = np.asarray(eval_start_idx).tolist()

    cost_latent = np.empty((n_eval, n_cand), dtype=np.float32)
    cost_cube = np.empty((n_eval, n_cand), dtype=np.float32)
    cost_gripper = np.empty((n_eval, n_cand), dtype=np.float32)

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    start_time = time.time()
    eval_world = world
    with autocast_ctx, torch.inference_mode():
        for batch_idx, start in enumerate(range(0, n_eval, batch_size)):
            end = min(start + batch_size, n_eval)
            chunk_n = end - start
            if chunk_n != eval_world.num_envs:
                eval_world.close()
                eval_world = build_world(cfg, chunk_n, plan_len)
            print(
                f'[eval] batch {batch_idx + 1}/{n_batches} '
                f'({start}:{end} of {n_eval})'
            )

            init_state, goal_state, _ = _extract_init_goal(
                dataset,
                eval_episodes_list[start:end],
                eval_start_list[start:end],
                goal_offset,
            )
            setup_from_dataset(eval_world, init_state, goal_state, callables)
            info = prep._prepare_info(eval_world.infos)

            candidates = torch.randn(
                chunk_n,
                n_cand,
                horizon,
                blocked_dim,
                generator=gen,
                device=device,
                dtype=dtype,
            )
            candidates = candidates * var_scale
            candidates[:, 0] = 0

            cost_latent[start:end] = latent_costs(
                model, info, candidates, device, dtype
            )
            env_actions = blocked_to_env_actions(
                candidates,
                process,
                action_dim,
                action_block,
                eval_world.envs.single_action_space,
            )

            goal_c = goal_cube[start:end]
            goal_g = goal_gripper[start:end]
            for j in range(n_cand):
                if j == 0 or (j + 1) % 10 == 0 or j + 1 == n_cand:
                    print(f'[eval]   candidate {j + 1}/{n_cand}')
                setup_from_dataset(
                    eval_world, init_state, goal_state, callables
                )
                execute_plan(eval_world, env_actions[:, j])
                cost_cube[start:end, j] = np.linalg.norm(
                    live_cube_positions(eval_world) - goal_c, axis=-1
                )
                cost_gripper[start:end, j] = np.linalg.norm(
                    live_gripper_positions(eval_world) - goal_g, axis=-1
                )
    elapsed = time.time() - start_time
    print(f'[eval] collection finished in {elapsed:.1f}s')

    output_dir = getattr(cfg.eval, 'output_dir', 'data')
    npz_path = (
        Path(hydra.utils.get_original_cwd())
        / output_dir
        / f'plan_{cfg.eval.name}.npz'
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        cost_latent=cost_latent,
        cost_cube=cost_cube,
        cost_gripper=cost_gripper,
        episode_idx=np.asarray(eval_episodes).reshape(n_eval).astype(np.int64),
        start_step=np.asarray(eval_start_idx).reshape(n_eval).astype(np.int32),
        cube_displacement=cube_displacement.astype(np.float32),
        arm_displacement=arm_displacement.astype(np.float32),
        horizon=np.int32(horizon),
        action_block=np.int32(action_block),
        var_scale=np.float32(var_scale),
        seed=np.int64(cfg.seed),
    )
    print(f'[eval] saved {n_eval}x{n_cand} candidate records to {npz_path.resolve()}')


if __name__ == '__main__':
    run()

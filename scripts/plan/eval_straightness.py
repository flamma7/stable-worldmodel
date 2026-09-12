"""
python scripts/plan/eval_straightness.py policy=quentinll/lewm-cube eval.name=ogb_cube_straight eval.dataset_name=galilai-group/ogb_cube_single seed=42 eval.num_eval=2000 eval.batch_size=50 -cn cube
"""

"""Encode consecutive latent triplets and save them to ``.npz``.

``eval.num_eval`` is the number of
dataset start frames. ``seed`` selects which starts. For each start,
frames ``t, t+1, t+2`` are encoded with the world-model encoder.

Writes ``<eval.output_dir>/<eval.name>.npz`` with:

- z_t, z_tp1, z_tp2: (N, D)
- episode_idx, start_step, seed
"""

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


TRIPLET_LEN = 3


def dataset_columns(dataset):
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return names


def episode_col(dataset):
    names = dataset_columns(dataset)
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


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


def sample_eval_rows(dataset, cfg, n_samples):
    col_name = episode_col(dataset)
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - TRIPLET_LEN
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
    chosen = g.choice(len(valid_indices), size=n_samples, replace=False)
    rows = np.sort(valid_indices[chosen])
    episodes = dataset.get_col_data(col_name)[rows]
    start_idx = dataset.get_col_data('step_idx')[rows]
    if len(episodes) < n_samples:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )
    return rows, episodes, start_idx


def _pixels_hwc(pixels):
    """``load_chunk`` pixels as (T, H, W, C) numpy."""
    if torch.is_tensor(pixels):
        pix = pixels.detach()
        if pix.ndim == 4 and pix.shape[1] in (1, 3):
            pix = pix.permute(0, 2, 3, 1)
        return pix.cpu().numpy()
    pix = np.asarray(pixels)
    if pix.ndim == 4 and pix.shape[1] in (1, 3):
        pix = np.transpose(pix, (0, 2, 3, 1))
    return pix


def load_pixel_triplets(dataset, episodes, start_steps):
    """Load frames t, t+1, t+2 as (B, 3, H, W, C) uint8."""
    starts = np.asarray(start_steps)
    chunk = dataset.load_chunk(
        np.asarray(episodes), starts, starts + TRIPLET_LEN
    )
    pixels = np.stack([_pixels_hwc(ep['pixels']) for ep in chunk], axis=0)
    if pixels.shape[1] != TRIPLET_LEN:
        raise ValueError(
            f'Expected {TRIPLET_LEN} frames per sample, got {pixels.shape[1]}.'
        )
    return pixels


def _encode_pixels(model, info):
    """Observation embedding only. Matches LeWM/PLDM encode used in plan."""
    obs = dict(info)
    obs.pop('action', None)
    obs.pop('act_emb', None)
    return model.encode(obs)


def encode_triplets(model, info, device, dtype):
    """Encode (B, 3, ...) pixels to (B, 3, D) latents."""
    prepared = {}
    for k, v in info.items():
        if torch.is_tensor(v):
            target_dtype = dtype if v.is_floating_point() else None
            prepared[k] = v.to(device=device, dtype=target_dtype)
    prepared.pop('emb', None)
    prepared.pop('feat', None)
    out = _encode_pixels(model, prepared)
    if isinstance(out, dict):
        z = out['emb']
    else:
        z = out
    if z.ndim != 3 or z.shape[1] != TRIPLET_LEN:
        raise ValueError(
            f'Expected encode emb (B, {TRIPLET_LEN}, D), got {tuple(z.shape)}.'
        )
    return z.detach().float().cpu().numpy()


@hydra.main(version_base=None, config_path='./config', config_name='cube')
def run(cfg: DictConfig):
    policy_name = cfg.get('policy', 'random')
    if policy_name == 'random':
        raise ValueError(
            'eval_straightness.py requires a world-model checkpoint '
            '(policy=<run-or-hf-id>), not policy=random.'
        )

    n_samples = int(cfg.eval.num_eval)
    batch_size = min(int(cfg.eval.get('batch_size', n_samples)), n_samples)
    n_batches = (n_samples + batch_size - 1) // batch_size

    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {'pixels': img_transform(cfg, img_dtype)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    prep = swm.policy.BasePolicy()
    prep.transform = transform

    rows, eval_episodes, eval_start_idx = sample_eval_rows(
        dataset, cfg, n_samples
    )
    print(rows)
    print(
        f'[eval] {n_samples} triplets, {batch_size} per encode batch '
        f'({n_batches} batches)'
    )

    drop = (
        ('motion_encoder',)
        if cfg.get('drop_motion_encoder', True)
        else None
    )
    model = swm.wm.utils.load_pretrained(cfg.policy, drop_modules=drop)
    if not hasattr(model, 'encode'):
        raise TypeError(
            'Loaded checkpoint has no encode; cannot compute z triplets.'
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

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    eval_episodes_list = np.asarray(eval_episodes).tolist()
    eval_start_list = np.asarray(eval_start_idx).tolist()

    z_all = []
    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    start_time = time.time()
    with autocast_ctx, torch.inference_mode():
        for batch_idx, start in enumerate(range(0, n_samples, batch_size)):
            end = min(start + batch_size, n_samples)
            print(
                f'[eval] batch {batch_idx + 1}/{n_batches} '
                f'({start}:{end} of {n_samples})'
            )
            pixels = load_pixel_triplets(
                dataset,
                eval_episodes_list[start:end],
                eval_start_list[start:end],
            )
            info = prep._prepare_info({'pixels': pixels})
            z_all.append(encode_triplets(model, info, device, dtype))
    elapsed = time.time() - start_time
    print(f'[eval] encoding finished in {elapsed:.1f}s')

    z = np.concatenate(z_all, axis=0)
    output_dir = getattr(cfg.eval, 'output_dir', 'data')
    npz_path = (
        Path(hydra.utils.get_original_cwd())
        / output_dir
        / f'{cfg.eval.name}.npz'
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        z_t=z[:, 0],
        z_tp1=z[:, 1],
        z_tp2=z[:, 2],
        episode_idx=np.asarray(eval_episodes).reshape(n_samples).astype(np.int64),
        start_step=np.asarray(eval_start_idx).reshape(n_samples).astype(np.int32),
        seed=np.int64(cfg.seed),
    )
    print(
        f'[eval] saved {n_samples} triplets {z.shape} to {npz_path.resolve()}'
    )


if __name__ == '__main__':
    run()

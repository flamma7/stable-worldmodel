import os
import time
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import VISReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained
from stable_worldmodel.wm.sigreg_diagnostics import SIGRegCollapseMonitor


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


_COMPILE_ATTRS = ('encoder', 'predictor')
LATEST_CKPT = 'latest.ckpt'
CONFIG_YAML = 'config.yaml'
# Launch-only keys: never overwritten by a restored training config.
_RESUME_KEEP = ('load_checkpoint_hf', 'hf')


def compile_lewm(model):
    """Compile the ViT-sized submodules. Leaves VISReg and small MLPs eager."""
    os.environ.setdefault('TORCHINDUCTOR_FX_GRAPH_CACHE', '1')
    for name in _COMPILE_ATTRS:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        setattr(model, name, torch.compile(mod))
    return model


def _hf_path_prefix(hf_cfg, run_name):
    """Repo-relative prefix, e.g. ``tdv/lewm_visreg/``.

    ``path_prefix`` already interpolates ``output_model_name`` in the default
    launcher config. If it does not, ``run_name`` is appended so the on-repo
    path is ``{path_prefix}/{output_model_name}/``.
    """
    prefix = str((hf_cfg or {}).get('path_prefix') or '').strip('/')
    name = run_name or ''
    if name and not prefix.endswith(name):
        prefix = f'{prefix}/{name}' if prefix else name
    return f'{prefix}/' if prefix else ''


def download_hf_file(hf_cfg, run_name, filename):
    """Download ``{repo_id}/{path_prefix}/{output_model_name}/{filename}``."""
    from huggingface_hub import hf_hub_download

    repo_id = hf_cfg['repo_id']
    repo_file = f'{_hf_path_prefix(hf_cfg, run_name)}{filename}'
    token = os.environ.get('HF_TOKEN') or hf_cfg.get('token')
    path = hf_hub_download(
        repo_id=repo_id,
        filename=repo_file,
        repo_type='model',
        token=token,
    )
    print(f'Downloaded HF {repo_id}/{repo_file} -> {path}')
    return path


def _cfg_from_checkpoint(ckpt_path):
    try:
        blob = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    except TypeError:
        blob = torch.load(str(ckpt_path), map_location='cpu')
    raw = blob.get('lewm_full_cfg')
    if raw is None:
        return None, blob
    return OmegaConf.create(raw), blob


def download_hf_resume_bundle(hf_cfg, run_name):
    """Fetch ``latest.ckpt`` and the matching training ``config.yaml``."""
    ckpt_path = str(Path(download_hf_file(hf_cfg, run_name, LATEST_CKPT)).resolve())
    saved_cfg = None
    try:
        yaml_path = download_hf_file(hf_cfg, run_name, CONFIG_YAML)
        saved_cfg = OmegaConf.load(yaml_path)
        print(f'Restored training config from HF {CONFIG_YAML}')
    except Exception as exc:
        print(f'No HF {CONFIG_YAML} ({exc}); trying checkpoint embedding')
        saved_cfg, blob = _cfg_from_checkpoint(ckpt_path)
        if saved_cfg is not None:
            print('Restored training config from checkpoint lewm_full_cfg')
        else:
            resume = (blob or {}).get('lewm_resume') or {}
            print(
                'WARNING: no saved config.yaml; using current hydra cfg. '
                f'Checkpoint meta: {resume}'
            )
    return ckpt_path, saved_cfg


def merge_saved_train_cfg(cfg, saved_cfg):
    """Replace training cfg with the saved snapshot; keep resume/HF launch keys."""
    keep = {key: OmegaConf.select(cfg, key) for key in _RESUME_KEEP}
    launch_max_epochs = cfg.trainer.max_epochs
    launch_wandb_enabled = OmegaConf.select(cfg, 'wandb.enabled')
    merged = OmegaConf.merge(cfg, saved_cfg)
    with open_dict(merged):
        for key, value in keep.items():
            if value is not None:
                merged[key] = value
        # Allow raising max_epochs on the resume job; never shrink it.
        if launch_max_epochs is not None:
            merged.trainer.max_epochs = max(
                int(launch_max_epochs), int(merged.trainer.max_epochs)
            )
        if launch_wandb_enabled is not None and merged.get('wandb') is not None:
            merged.wandb.enabled = launch_wandb_enabled
        merged.load_checkpoint_hf = True
    print(
        'Merged saved training config '
        f'(max_epochs={merged.trainer.max_epochs}, '
        f'lr={merged.optimizer.lr}, seed={merged.seed})'
    )
    return merged


def _swap_compiled_children(model, restore=None):
    """Temporarily unwrap OptimizedModule children for a loadable state_dict."""
    if restore is not None:
        for name, compiled in restore:
            setattr(model, name, compiled)
        return None
    swaps = []
    for name, child in list(model.named_children()):
        orig = getattr(child, '_orig_mod', None)
        if orig is not None:
            setattr(model, name, orig)
            swaps.append((name, child))
    return swaps


def _compiled_module_prefixes(module):
    """Dotted names of submodules wrapped by ``torch.compile`` (longest first)."""
    prefixes = [
        name
        for name, child in module.named_modules()
        if name and getattr(child, '_orig_mod', None) is not None
    ]
    prefixes.sort(key=len, reverse=True)
    return prefixes


def _align_state_dict_to_module(module, state_dict):
    """Map checkpoint keys onto the live module (eager <-> ``_orig_mod``)."""
    prefixes = _compiled_module_prefixes(module)
    aligned = {}
    for key, value in state_dict.items():
        new_key = key.replace('._orig_mod', '')
        for prefix in prefixes:
            eager = f'{prefix}.'
            compiled = f'{prefix}._orig_mod.'
            if new_key.startswith(eager):
                new_key = compiled + new_key[len(eager) :]
                break
        aligned[new_key] = value
    return aligned


def _install_compile_ckpt_hook(pl_module):
    """Remap keys in LightningModule.on_load_checkpoint, which runs before load."""
    previous = getattr(pl_module, 'on_load_checkpoint', None)

    def on_load_checkpoint(checkpoint):
        resume = checkpoint.get('lewm_resume') or {}
        loops = checkpoint.get('loops') or {}
        fit_progress = (
            (loops.get('fit_loop') or {}).get('epoch_progress')
            if isinstance(loops.get('fit_loop'), dict)
            else None
        )
        print(
            'Lightning checkpoint restore: '
            f"epoch={checkpoint.get('epoch')} "
            f"global_step={checkpoint.get('global_step')} "
            f"has_loops={'loops' in checkpoint} "
            f"has_optim={'optimizer_states' in checkpoint} "
            f"has_schedulers={'lr_schedulers' in checkpoint} "
            f"lewm_resume={resume} fit_loop.epoch_progress={fit_progress}"
        )
        sd = checkpoint.get('state_dict')
        if isinstance(sd, dict):
            checkpoint['state_dict'] = _align_state_dict_to_module(
                pl_module, sd
            )
        if callable(previous):
            previous(checkpoint)

    pl_module.on_load_checkpoint = on_load_checkpoint


class WallClockThroughput(Callback):
    """Wall-clock samples/sec via time.perf_counter, no CUDA sync."""

    def __init__(self):
        super().__init__()
        self._t0 = None
        self._n_samples = 0
        self._n_batches = 0

    def on_train_start(self, trainer, pl_module):
        self._reset()

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._n_samples += self._global_batch_size(trainer, batch)
        self._n_batches += 1

        log_every = trainer.log_every_n_steps or 1
        if (batch_idx + 1) % log_every != 0:
            return

        now = time.perf_counter()
        dt = now - self._t0
        if dt > 0 and self._n_batches > 0:
            log_kw = dict(on_step=True, on_epoch=False, rank_zero_only=True)
            pl_module.log(
                'trainer/samples_per_sec', self._n_samples / dt, **log_kw
            )
            pl_module.log(
                'trainer/batches_per_sec', self._n_batches / dt, **log_kw
            )
            pl_module.log(
                'trainer/batch_time_sec', dt / self._n_batches, **log_kw
            )
        self._t0 = now
        self._n_samples = 0
        self._n_batches = 0

    def _reset(self):
        self._t0 = time.perf_counter()
        self._n_samples = 0
        self._n_batches = 0

    @staticmethod
    def _global_batch_size(trainer, batch):
        if isinstance(batch, dict):
            for key in ('pixels', 'action'):
                if key in batch and torch.is_tensor(batch[key]):
                    local = batch[key].shape[0]
                    break
            else:
                local = next(
                    v.shape[0] for v in batch.values() if torch.is_tensor(v)
                )
        elif torch.is_tensor(batch):
            local = batch.shape[0]
        else:
            local = len(batch)
        return local * trainer.world_size


class _VisregDiagClose(Callback):
    def on_train_end(self, trainer, pl_module):
        diag = getattr(pl_module, 'visreg_diagnostics', None)
        if diag is not None:
            diag.close(pl_module)


class SaveCkptCallback(Callback):
    """Save epoch weights plus a Lightning ``latest.ckpt`` for resume."""

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval: int = 1,
        hf_cfg=None,
        full_cfg=None,
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.full_cfg = full_cfg
        self.epoch_interval = epoch_interval
        self.hf_cfg = hf_cfg or {}
        self._hf_api = None
        self._logged_resume = False

    @property
    def _ckpt_dir(self):
        return (
            swm.data.utils.get_cache_dir(sub_folder='checkpoints')
            / self.run_name
        )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['lewm_resume'] = {
            'epoch': int(trainer.current_epoch),
            'global_step': int(trainer.global_step),
            'max_epochs': int(trainer.max_epochs)
            if trainer.max_epochs is not None
            else None,
        }
        if self.full_cfg is not None:
            checkpoint['lewm_full_cfg'] = OmegaConf.to_container(
                self.full_cfg, resolve=True
            )

    def on_train_epoch_start(self, trainer, pl_module):
        if self._logged_resume:
            return
        self._logged_resume = True
        progress = trainer.fit_loop.epoch_progress.current
        print(
            'Trainer state at first epoch start: '
            f'current_epoch={trainer.current_epoch} '
            f'global_step={trainer.global_step} '
            f'max_epochs={trainer.max_epochs} '
            f'progress(ready={progress.ready} started={progress.started} '
            f'processed={progress.processed} completed={progress.completed})'
        )

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch + 1
        should_save = (
            epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs
        )
        if should_save:
            self._save(pl_module, trainer, epoch)

    def _get_hf_api(self):
        if self._hf_api is not None:
            return self._hf_api

        import logging
        from huggingface_hub import HfApi
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
        token = os.environ.get('HF_TOKEN') or self.hf_cfg.get('token')
        self._hf_api = HfApi(token=token)
        return self._hf_api

    def _upload_to_hf(self, weights_file):
        if not self.hf_cfg.get('enabled'):
            return

        repo_id = self.hf_cfg['repo_id']
        prefix = _hf_path_prefix(self.hf_cfg, self.run_name)

        hf_api = self._get_hf_api()
        hf_api.upload_file(
            path_or_fileobj=str(self._ckpt_dir / weights_file),
            path_in_repo=f'{prefix}{weights_file}',
            repo_id=repo_id,
            repo_type='model',
        )
        config_path = self._ckpt_dir / 'config.json'
        if config_path.exists():
            hf_api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo=f'{prefix}config.json',
                repo_id=repo_id,
                repo_type='model',
            )
        print(f'Uploaded {prefix}{weights_file} to {repo_id}')

    def _save_full_config(self):
        if self.full_cfg is None:
            return None
        path = self._ckpt_dir / CONFIG_YAML
        OmegaConf.save(
            OmegaConf.create(
                OmegaConf.to_container(self.full_cfg, resolve=True)
            ),
            path,
        )
        return CONFIG_YAML

    def _save(self, pl_module, trainer, epoch):
        model = pl_module.model
        filename = f'weights_epoch_{epoch}.pt'
        latest_path = self._ckpt_dir / LATEST_CKPT
        # save_checkpoint must run on every rank (Lightning barrier).
        swaps = _swap_compiled_children(model)
        try:
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(latest_path))
            if trainer.is_global_zero:
                save_pretrained(
                    model,
                    run_name=self.run_name,
                    config=self.cfg,
                    filename=filename,
                )
                self._save_full_config()
                self._upload_to_hf(filename)
                self._upload_to_hf(LATEST_CKPT)
                if (self._ckpt_dir / CONFIG_YAML).exists():
                    self._upload_to_hf(CONFIG_YAML)
        finally:
            _swap_compiled_children(model, restore=swaps)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    # Extract the context embeddings and actions
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM + VISReg loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output.update(self.visreg(emb.transpose(0, 1)))
    output['loss'] = output['pred_loss'] + output['visreg_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=False) # changed to false for faster ddp training

    diag = getattr(self, 'visreg_diagnostics', None)
    if diag is not None and stage == 'fit':
        diag.update(emb, self)
    return output


def _visreg_from_cfg(cfg):
    visreg_cfg = OmegaConf.to_container(cfg.loss.visreg, resolve=True)
    visreg_cfg.pop('diagnostics', None)
    return VISReg(**visreg_cfg)


@hydra.main(version_base=None, config_path='./config', config_name='lewm_visreg')
def run(cfg):
    hf_cfg = cfg.get('hf')
    if hf_cfg is not None:
        hf_cfg = OmegaConf.to_container(hf_cfg, resolve=True)

    ckpt_path = None
    if cfg.get('load_checkpoint_hf'):
        if not hf_cfg or not hf_cfg.get('repo_id'):
            raise ValueError(
                'load_checkpoint_hf=true requires hf.repo_id '
                '(see launcher config)'
            )
        ckpt_path, saved_cfg = download_hf_resume_bundle(
            hf_cfg, cfg.output_model_name
        )
        if saved_cfg is not None:
            cfg = merge_saved_train_cfg(cfg, saved_cfg)
            if cfg.get('hf') is not None:
                hf_cfg = OmegaConf.to_container(cfg.hf, resolve=True)

    if cfg.get('make_it_fast', False):
        spt.make_it_fast()

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.val_loader)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    if cfg.get('compile', True):
        world_model = compile_lewm(world_model)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        visreg=_visreg_from_cfg(cfg),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )
    _install_compile_ckpt_hook(world_model)

    diag_cfg = cfg.loss.visreg.get('diagnostics')
    if diag_cfg is not None:
        diag_cfg = OmegaConf.to_container(diag_cfg, resolve=True)
        if diag_cfg.get('enabled', True):
            world_model.visreg_diagnostics = SIGRegCollapseMonitor(diag_cfg)

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        wandb_kwargs = OmegaConf.to_container(cfg.wandb.config, resolve=True)
        if cfg.wandb.get('disable_system', True):
            import wandb

            wandb_kwargs['settings'] = wandb.Settings(x_disable_stats=True)
        logger = WandbLogger(**wandb_kwargs)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    hf_cfg = cfg.get('hf')
    if hf_cfg is not None:
        hf_cfg = OmegaConf.to_container(hf_cfg, resolve=True)

    save_ckpt_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
        hf_cfg=hf_cfg,
        full_cfg=cfg,
    )
    callbacks = [save_ckpt_callback]
    if cfg.get('log_throughput', True):
        callbacks.append(WallClockThroughput())
    if getattr(world_model, 'visreg_diagnostics', None) is not None:
        callbacks.append(_VisregDiagClose())

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    if ckpt_path is None:
        legacy = run_dir / f'{cfg.output_model_name}_weights.ckpt'
        if legacy.exists():
            ckpt_path = str(legacy.resolve())

    manager_kwargs = dict(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path,
        seed=cfg.seed,
    )
    # Full trainer resume (epoch / optim / schedulers / RNG), not weights-only.
    if ckpt_path is not None:
        manager_kwargs['weights_only'] = False

    manager = spt.Manager(**manager_kwargs)

    manager()
    return


if __name__ == '__main__':
    run()

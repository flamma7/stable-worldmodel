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
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


_COMPILE_ATTRS = ('encoder', 'predictor', 'motion_encoder')


def compile_lewm(model):
    """Compile the ViT-sized submodules. Leaves SIGReg and small MLPs eager."""
    os.environ.setdefault('TORCHINDUCTOR_FX_GRAPH_CACHE', '1')
    for name in _COMPILE_ATTRS:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        setattr(model, name, torch.compile(mod))
    return model


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


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1, hf_cfg=None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.hf_cfg = hf_cfg or {}
        self._hf_api = None

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

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
        ckpt_dir = (
            swm.data.utils.get_cache_dir(sub_folder='checkpoints')
            / self.run_name
        )
        prefix = self.hf_cfg.get('path_prefix') or self.run_name
        if prefix and not prefix.endswith('/'):
            prefix = f'{prefix}/'

        hf_api = self._get_hf_api()
        hf_api.upload_file(
            path_or_fileobj=str(ckpt_dir / weights_file),
            path_in_repo=f'{prefix}{weights_file}',
            repo_id=repo_id,
            repo_type='model',
        )
        config_path = ckpt_dir / 'config.json'
        if config_path.exists():
            hf_api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo=f'{prefix}config.json',
                repo_id=repo_id,
                repo_type='model',
            )
        print(f'Uploaded {prefix}{weights_file} to {repo_id}')

    def _save(self, model, epoch):
        filename = f'weights_epoch_{epoch}.pt'
        swaps = _swap_compiled_children(model)
        try:
            save_pretrained(
                model,
                run_name=self.run_name,
                config=self.cfg,
                filename=filename,
            )
            self._upload_to_hf(filename)
        finally:
            _swap_compiled_children(model, restore=swaps)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    alpha = cfg.loss.tdv.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']
    feat = output['feat']  # (B, T, N, D) H_t = [CLS + patches]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # TDV: Δx_t = x_{t+1} - x_t, δz_t = m(Δx_t, H_t)
    # L_TDV = ||z_t + δz_t - z_{t+1}||^2
    delta_x = batch['pixels'][:, 1:] - batch['pixels'][:, :-1]
    delta_z = self.model.predict_delta(delta_x, feat[:, :-1])

    # LeWM + TDV loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['tdv_loss'] = (emb[:, :-1] + delta_z - emb[:, 1:]).pow(2).mean()
    output['loss'] = (
        output['pred_loss']
        + lambd * output['sigreg_loss']
        + alpha * output['tdv_loss']
    )

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=False) # changed to false for faster ddp training
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm_tdv')
def run(cfg):
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
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
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
    )
    callbacks = [save_ckpt_callback]
    if cfg.get('log_throughput', True):
        callbacks.append(WallClockThroughput())

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == '__main__':
    run()

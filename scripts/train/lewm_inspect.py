import os
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
from stable_worldmodel.wm.utils import load_pretrained
from stable_worldmodel.wm.sigreg_diagnostics import SIGRegCollapseMonitor


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


_COMPILE_ATTRS = ('encoder', 'predictor')


def compile_lewm(model):
    """Compile the ViT-sized submodules. Leaves SIGReg and small MLPs eager."""
    os.environ.setdefault('TORCHINDUCTOR_FX_GRAPH_CACHE', '1')
    for name in _COMPILE_ATTRS:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        setattr(model, name, torch.compile(mod))
    return model


class _InspectTick(torch.nn.Module):
    """Dummy parameter so Lightning can tick ``global_step`` with no WM updates."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class FrozenEvalCallback(Callback):
    """Keep eval mode so BN running stats and dropout stay frozen."""

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.eval()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.eval()


class _SigregDiagClose(Callback):
    def on_train_end(self, trainer, pl_module):
        diag = getattr(pl_module, 'sigreg_diagnostics', None)
        if diag is not None:
            diag.close(pl_module)


def resolve_pretrained(model_dir: str) -> str:
    """Resolve a folder, ``.pt`` file, cache-relative path, or HF repo id.

    Search order: explicit path, launch-dir walk for
    ``.stablewm/checkpoints/<name>``, then ``load_pretrained`` (cache / HF).
    """
    raw = str(model_dir).strip()
    path = Path(raw).expanduser()
    if path.exists():
        return str(path.resolve())

    try:
        start = Path(hydra.utils.get_original_cwd()).resolve()
    except Exception:
        start = Path.cwd().resolve()

    rel = Path(raw)
    for base in [start, *start.parents]:
        for candidate in (base / rel, base / '.stablewm' / 'checkpoints' / rel):
            if candidate.exists():
                return str(candidate.resolve())
    return raw


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, log losses. No WM gradients."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    with torch.no_grad():
        output = self.model.encode(batch)

        emb = output['emb']  # (B, T, D)
        act_emb = output['act_emb']

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]

        tgt_emb = emb[:, n_preds:]
        pred_emb = self.model.predict(ctx_emb, ctx_act)

        output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
        output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
        logged = output['pred_loss'] + lambd * output['sigreg_loss']

    # Dummy graph so Lightning's manual backward / optimizer.step tick
    # global_step without touching world-model weights.
    tick = self.inspect_tick.weight.to(device=logged.device, dtype=logged.dtype)
    output['loss'] = logged + tick.sum() * 0

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(
        losses_dict, on_step=True, sync_dist=False, prog_bar=True
    )

    diag = getattr(self, 'sigreg_diagnostics', None)
    if diag is not None and stage == 'fit':
        diag.update(emb, self)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm_inspect')
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
        f'Loading dataset "{dataset_name}" from '
        f'{"local cache: " + cache_dir if cache_dir else "default location"}'
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

    pretrained_name = resolve_pretrained(cfg.pretrained_model_dir)
    print(f'Loading frozen LeWM from {pretrained_name}')
    world_model = load_pretrained(pretrained_name)
    if getattr(world_model, 'motion_encoder', None) is not None:
        print(
            'WARNING: checkpoint has a motion_encoder; '
            'inspect forward is vanilla LeWM (no TDV).'
        )
    world_model.requires_grad_(False)
    world_model.eval()
    n_params = sum(p.numel() for p in world_model.parameters())
    n_train = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
    print(f'Loaded LeWM: {n_params:,} params, {n_train:,} trainable')

    if cfg.get('compile', False):
        world_model = compile_lewm(world_model)

    data_module = spt.data.DataModule(train=train, val=val)
    pl_module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        inspect_tick=_InspectTick(),
        forward=partial(lejepa_forward, cfg=cfg),
        # Dummy SGD (lr=0) on inspect_tick only: ticks global_step, no WM updates.
        optim={
            'inspect_tick': {
                'modules': 'inspect_tick',
                'optimizer': {'type': 'SGD', 'lr': 0.0},
                'interval': 'step',
            }
        },
    )
    pl_module.model.requires_grad_(False)

    diag_cfg = cfg.loss.sigreg.get('diagnostics')
    if diag_cfg is not None:
        diag_cfg = OmegaConf.to_container(diag_cfg, resolve=True)
        if diag_cfg.get('enabled', True):
            pl_module.sigreg_diagnostics = SIGRegCollapseMonitor(diag_cfg)

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

    callbacks = [FrozenEvalCallback()]
    if getattr(pl_module, 'sigreg_diagnostics', None) is not None:
        callbacks.append(_SigregDiagClose())

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=False,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=pl_module,
        data=data_module,
        seed=cfg.seed,
    )
    manager()
    return


if __name__ == '__main__':
    run()

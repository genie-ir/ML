import os
from tqdm import tqdm
from os import makedirs, rename, getenv
from os.path import join, exists
import torch
import torchvision
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from omegaconf import OmegaConf
from libs.basicIO import signal_save
from torch import Tensor
from datetime import timedelta
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.rank_zero import rank_zero_only 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint as ModelCheckpointBasic, Callback, LearningRateMonitor

class ModelCheckpointBase(ModelCheckpointBasic):
    def __init__(self, dirpath = None, filename: Optional[str] = None, monitor: Optional[str] = None, verbose: bool = False, save_last: Optional[bool] = None, save_top_k: int = 1, save_weights_only: bool = False, mode: str = "min", auto_insert_metric_name: bool = True, every_n_train_steps: Optional[int] = None, train_time_interval: Optional[timedelta] = None, every_n_epochs: Optional[int] = None, save_on_train_epoch_end: Optional[bool] = None, lastname='last', lastname_format=''):
        self.CHECKPOINT_NAME_LAST = lastname_format + lastname # this line it should be before super.init | Example: lastname_format='{epoch}-{step}'
        super().__init__(dirpath, filename, monitor, verbose, save_last, save_top_k, save_weights_only, mode, auto_insert_metric_name, every_n_train_steps, train_time_interval, every_n_epochs, save_on_train_epoch_end)

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]):
        # checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object
        # checkpoint['genie'] = checkpoint.get('genie', {}) 
        # key = 'best_of_{}'.format(str(self.monitor))
        # checkpoint['genie'][key] = self.current_score
        pass

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]):
        # my_cool_pickable_object = checkpoint['something_cool_i_want_to_save']
        pass


class SetupCallbackBase(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            makedirs(self.logdir, exist_ok=True)
            makedirs(self.ckptdir, exist_ok=True)
            makedirs(self.cfgdir, exist_ok=True)

            OmegaConf.save(self.config, join(self.cfgdir, 'project.yaml'))
            OmegaConf.save(OmegaConf.create({'lightning': self.lightning_config}), join(self.cfgdir, 'lightning.yaml'))
            # OmegaConf.save(self.config, join(self.cfgdir, '{}-project.yaml'.format(self.now)))
            # OmegaConf.save(OmegaConf.create({'lightning': self.lightning_config}), join(self.cfgdir, '{}-lightning.yaml'.format(self.now)))
            logger.info('Project config: {}'.format(self.config))
            logger.info('Lightning config: {}'.format(self.lightning_config))
        else:
            logger.warning('vvvvvv on_fit_start > gRank -> {}'.format(trainer.global_rank))
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and exists(self.logdir):
                logger.warning('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
                dst, name = os.path.split(self.logdir)
                dst = join(dst, 'child_runs', name)
                makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class CustomProgressBarBase(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar

    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop('v_num', None)
        self._items_dict = items
        # logger.debug('items={}'.format(items))
        return items


class SignalLoggerBase(Callback):
    """
        In this function in sometimes we call `Image` instead `Signal` but ther is no limitation
    """
    def __init__(self, batch_frequency, max_signals, clamp=True, increase_log_steps=True, use_log_local_fn=True, opt_params=None, nowname='', ignoreFlag_on_train_batch_end=False, ignoreFlag_on_validation_batch_end=False):
        super().__init__()
        self.ignoreFlag_on_train_batch_end = bool(ignoreFlag_on_train_batch_end)
        self.ignoreFlag_on_validation_batch_end = bool(ignoreFlag_on_validation_batch_end)
        self.nowname = nowname
        self.batch_freq = batch_frequency
        self.max_signals = max_signals
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.use_log_local_fn = use_log_local_fn
        self.opt_params = opt_params if opt_params else dict()

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise NotImplementedError
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f'{split}/{k}'] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        raise NotImplementedError
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f'{split}/{k}'
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def _tb(self, pl_module, images, batch_idx, split):
        raise NotImplementedError
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f'{split}/{k}'
            # pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
    
    @rank_zero_only
    def _genie(self, pl_module, images, batch_idx, split):
        return # Not statement becuse images already loged into `save_dir` By `self.log_local` function.

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        """task of this function is save signals into save_dir. Notic that (`save_dir` is address of `logdir`)"""
        root = join(save_dir, 'images', split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=self.opt_params.get('make_grid_nrow', 4)) # this grid finally contains table of iamges like this -> [images[k].shape[0]/nrow, nrow] ; Notic: grid is tensor with shape: ch x h? x w?
            grid = (grid+1.0)/2.0 # turn each image from range [-1,1] to range [0,1]; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = '{}_gs-{:06}_e-{:06}_b-{:06}.png'.format(k, global_step, current_epoch, batch_idx)
            path = join(root, filename)
            signal_save(grid, path)

    @rank_zero_only # DELETE
    def log_signal(self, pl_module, batch, batch_idx, split='train'):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, 'log_images') and
                callable(pl_module.log_images) and
                self.max_signals > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)
                if '_' in images:
                    value_of_underscore = images.pop('_')

            for k in images:
                # logger.critical(k)
                # logger.warning(images[k].shape)
                N = min(images[k].shape[0], self.max_signals)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.) # it garanty that signal is in range [-1, +1]

            if self.use_log_local_fn:
                save_to_dir = join(getenv('GENIE_ML_ROOT'), self.nowname, 'ImageLogger') #pl_module.logger.save_dir
                self.log_local(save_to_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx) # save images on disk.
            
            _logger = getattr(pl_module.logger, 'fn_name', '_tb')
            logger_log_images_fn = getattr(self, _logger, lambda *args, **kwargs: None) 
            logger_log_images_fn(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    @rank_zero_only # DELETE
    def check_frequency(self, batch_idx): #TODO!!!!!!!!!!!!!!
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): # ,dataloader_idx
        if self.ignoreFlag_on_train_batch_end:
            return
        self.log_signal(pl_module, batch, batch_idx, split='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx): # in this case outputs is same as pl_module!!
        if self.ignoreFlag_on_validation_batch_end:
            return
        self.log_signal(pl_module, batch, batch_idx, split='val')


class CBBase(Callback):
    """this callback defiend only for handling some BUGs of lightning and give more control to user"""
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): # ,dataloader_idx
        # logger.warning(trainer.logged_metrics)
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx): # in this case outputs is same as pl_module!!
        if False and (batch_idx % trainer.log_every_n_steps == 0):
            S = pl_module.current_epoch * trainer.num_val_batches[dataloader_idx] + batch_idx
            M = {k: v.item() for k, v in trainer.logged_metrics.items()}
            for tl in trainer.loggers:
                handiCall = getattr(tl, 'log_metrics_handiCall', lambda *args, **kwargs: None)
                handiCall(metrics={**M, f'val_dl_{dataloader_idx}': dataloader_idx}, step=S)
            logger.critical('logs -> {}'.format(trainer.logged_metrics))
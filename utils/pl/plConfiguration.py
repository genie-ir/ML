import torch
from os.path import join
from loguru import logger
import os, cowsay, sys, argparse
from libs.basicIO import pathBIO, puml, readBIO
from os import environ, getenv
from omegaconf import OmegaConf
from libs.dyimport import Import, instantiate_from_config as instantiate_from_config_base
from pytorch_lightning.trainer import Trainer

class ConfigBase:
    def __new__(cls, **kwargs):
        super().__new__(cls)
        return cls.fn(**kwargs)
    
    @classmethod
    def instantiate_from_config(cls, config, kwargs=dict()):
        return instantiate_from_config_base(config, kwargs=kwargs)

    @classmethod
    def __app_ctrl_cfg(cls, inpcfg):
        outcfg = dict()
        for inpcfg_k, inpcfg_v in inpcfg.items():
            if inpcfg_k.startswith('@') == False:
                if inpcfg_k.startswith('$'):
                    if bool(inpcfg_v):
                        outcfg[inpcfg_k[1:]] = eval('cls.vars["' + inpcfg_k[1:] + '"]') 
                else:
                    outcfg[inpcfg_k] = inpcfg_v
        return outcfg
    
    @classmethod
    def fn(cls, **kwargs):
        now = kwargs['now']
        opt = kwargs['opt']
        unknown = kwargs['unknown']
        cfgdir = kwargs['cfgdir']
        logdir = kwargs['logdir']
        ckptdir = kwargs['ckptdir']
        nowname = kwargs['nowname']

        cls.vars = {
            'now': now, 'opt': opt, 'unknown': unknown, 'cfgdir': cfgdir,
            'logdir': logdir, 'ckptdir': ckptdir, 'nowname': nowname
        }
        
        app_ctrl_cfg = cls.__app_ctrl_cfg
        app_name_master = opt.app.split(':')[0]
        app_cfg_master = {
            'callback': instantiate_from_config_base({'target': f'apps.{app_name_master}.configs.core.callback.yaml'}, kwargs={'dotdictFlag': False}) or dict()
        }

        if environ['GENIE_ML_ADD_TAG_TO_CACHEDIR'] == 'True':
            environ['GENIE_ML_CACHEDIR'] = join(environ['GENIE_ML_CACHEDIR'], nowname)
        nondefault_trainer_args = kwargs['nondefault_trainer_args']

        # configure model&trainer&data #####################################
        # init and save configs
        configs = []
        for _cfg in opt.base:
            if _cfg.lower().endswith('.yaml') or _cfg.lower().endswith('.json'):
                if not os.path.exists(_cfg):
                    _cfg2 = pathBIO('//apps/{}/configs/{}'.format(app_name_master, _cfg))
                    if os.path.exists(_cfg2):
                        cfg = _cfg2
                    else:
                        cowsay.cow('NotImplementedError:\nplease define `APP={} | config={}`'.format(app_name_master, _cfg))
                        sys.exit()
                else:
                    cfg = _cfg
            else:
                cfg = pathBIO('//apps/{}/configs/{}/.yaml'.format(app_name_master, _cfg))
                if not os.path.exists(cfg):
                    cowsay.cow('NotImplementedError:\nplease define `APP={} | NET={}`'.format(app_name_master, _cfg))
                    sys.exit()
            cfg_read = OmegaConf.create(readBIO(cfg, dotdictFlag=False, instantiate_from_config=instantiate_from_config_base))
            configs.append(cfg_read)
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop('lightning', OmegaConf.create())
        
        # print(type(config), config)
        # print('='*60)
        # print(lightning_config)
        # print('='*60)
        # input()
        # return
        
        # merge trainer cli with config
        trainer_config = lightning_config.get('trainer', OmegaConf.create())
        
        # correction phase on trainer_config
        trainer_config['distributed_backend'] = 'ddp' # default to ddp
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not 'gpus' in trainer_config:
            del trainer_config['distributed_backend']
            cpu = True
        else:
            gpuinfo = trainer_config['gpus']
            logger.info(f'Running on GPUs {gpuinfo}')
            cpu = False
        
        # data
        data = cls.instantiate_from_config(config.data)
        data.setup()
        
        dataTrainDataloader = data._train_dataloader()
        if dataTrainDataloader is None:
            log_every_n_steps_var = 0
        else:
            log_every_n_steps_var = int(len(dataTrainDataloader) / 10)
        if log_every_n_steps_var < 1:
            log_every_n_steps_var = 1
        trainer_config['log_every_n_steps'] = int(abs(trainer_config.get('log_every_n_steps', log_every_n_steps_var)))
        trainer_config['num_sanity_val_steps'] = trainer_config.get('num_sanity_val_steps', 0) # num validation batch that run befor training. default 2 I set to 0 as default.
        logger.info('trainer_config={}'.format(trainer_config))
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        if bool(str(opt.Rfn)):
            config.model['params']['Rfn'] = opt.Rfn
        
        config.model['params']['ckpt'] = str(opt.resume_from_checkpoint or config.model['params'].get('ckpt', ''))
        
        config.model['params']['set_default_tensor_type'] = str(config.model['params'].get('set_default_tensor_type', 'float32'))
        if config.model['params']['set_default_tensor_type'] == 'float64':
            torch.set_default_tensor_type(torch.DoubleTensor) # change default from torch.float32 -> torch.float64
        
        model = cls.instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # *****************[default logger]*****************
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            'wandb': {
                'target': 'pytorch_lightning.loggers.WandbLogger',
                'params': {**{
                    'name': nowname,
                    'save_dir': logdir,
                    'offline': opt.debug,
                    'id': nowname,
                }, **app_ctrl_cfg(app_cfg_master['callback'].get('wandb', dict()))}
            },
            'testtube': {
                'target': 'pytorch_lightning.loggers.TestTubeLogger',
                'params': {**{
                    'name': 'testtube',
                    'save_dir': logdir,
                }, **app_ctrl_cfg(app_cfg_master['callback'].get('testtube', dict()))}
            },
            'tensorboard': {
                'target': 'pytorch_lightning.loggers.TensorBoardLogger',
                'params': {**{
                    'name': nowname,
                    'save_dir': logdir,
                }, **app_ctrl_cfg(app_cfg_master['callback'].get('tensorboard', dict()))}
            },
            'genie': {
                'target': 'apps.' + app_name_master + '.modules.genie_logger.GenieLogger',
                'params': {**{
                    'name': nowname,
                    'save_dir': logdir,
                    'hash_ignore': opt.hash_ignore
                }, **app_ctrl_cfg(app_cfg_master['callback'].get('genie', dict()))}
            },
        }
         
        default_logger_cfg = default_logger_cfgs[opt.logger_ml] # default is: 'genie'
        logger_cfg = lightning_config.get('logger', OmegaConf.create()) # lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs['logger'] = cls.instantiate_from_config(logger_cfg)

        # *****************[model checkpoint]*****************
        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            'target': 'apps.' + app_name_master + '.modules.callback.ModelCheckpoint',
            'params': {**{
                'lastname': 'best',
                'every_n_epochs': 0,
                'dirpath': ckptdir,
                'verbose': True,
                'save_last': True
            }, **app_ctrl_cfg(app_cfg_master['callback'].get('bestCKPT', dict()))}
        }
        if app_cfg_master['callback']['bestCKPT'].get('@off', False) == False:
            assert 'monitor' in default_modelckpt_cfg['params'] and 'mode' in default_modelckpt_cfg['params']
        if hasattr(model, 'monitor'):
            logger.critical(f'||||Monitoring {model.monitor} as checkpoint metric.')
            default_modelckpt_cfg['params']['monitor'] = model.monitor
            default_modelckpt_cfg['params']['save_top_k'] = 3

        modelckpt_cfg = lightning_config.get('modelcheckpoint', OmegaConf.create()) # lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        _checkpoint_callback = cls.instantiate_from_config(modelckpt_cfg)
        logger.info('lightning_config.modelcheckpoint={}'.format( lightning_config.get('modelcheckpoint',{}) ))

        # *****************[sets up log directory]*****************
        # add callback which sets up log directory
        default_callbacks_cfg = {
            'setup_callback': {
                'target': 'apps.' + app_name_master + '.modules.callback.SetupCallback',
                'params': {**{
                    'resume': opt.resume,
                    'now': now,
                    'logdir': logdir,
                    'ckptdir': ckptdir,
                    'cfgdir': cfgdir,
                    'config': config,
                    'lightning_config': lightning_config,
                }, **app_ctrl_cfg(app_cfg_master['callback'].get('setup_callback', dict()))}
            },
            'custom_progressBar': {
                'target': 'apps.' + app_name_master + '.modules.callback.CustomProgressBar',
                'params': {
                    **app_ctrl_cfg(app_cfg_master['callback'].get('custom_progressBar', dict()))
                }
            },
            'signal_logger': {
                'target': 'apps.' + app_name_master + '.modules.callback.SignalLogger',
                'params': {**{
                    'clamp': True,
                    'nowname': nowname
                }, **app_ctrl_cfg(app_cfg_master['callback'].get('signal_logger', dict()))}
            },
            # 'learning_rate_logger': { # it must be uncomment!!!!!!
            #     'target': 'apps.' + app_name_master + '.modules.callback.LearningRateMonitor',
            #     'params': {**{
            #         'logging_interval': 'step',
            #         #'log_momentum': True
            #     }, **app_ctrl_cfg(app_cfg_master['callback'].get('learning_rate_logger', dict()))}
            # },
            'cb': {
                'target': 'apps.' + app_name_master + '.modules.callback.CB',
                'params': {
                    **app_ctrl_cfg(app_cfg_master['callback'].get('cb', dict()))
                }
            },
            'lastCKPT': {
                'target': 'apps.' + app_name_master + '.modules.callback.ModelCheckpoint',
                'params': {**{
                    'lastname': 'last',
                    'every_n_epochs': 0,
                    'dirpath': ckptdir,
                    'save_last': True,
                    'verbose': True
                }, **app_ctrl_cfg(app_cfg_master['callback'].get('lastCKPT', dict()))}
            },
        }
        if app_cfg_master['callback']['signal_logger'].get('@off', False) == False:
            assert 'batch_frequency' in default_callbacks_cfg['signal_logger']['params'] and 'max_signals' in default_callbacks_cfg['signal_logger']['params']
        
        app_cfg_master_callback_keys = list(app_cfg_master['callback'].keys())
        for newcb in app_cfg_master_callback_keys:
            if newcb.startswith('@'):
                newcb_name = newcb[1:]
                if not (newcb_name in default_callbacks_cfg):
                    if app_cfg_master['callback'][newcb].get('@off', False) == False:
                        default_callbacks_cfg[newcb_name] = {
                            'target': 'apps.' + app_name_master + f'.modules.callback.{newcb_name}',
                            'params': {
                                **app_ctrl_cfg(app_cfg_master['callback'].get(newcb, dict()))
                            }
                        }
        
        callbacks_cfg = lightning_config.get('callbacks', OmegaConf.create()) # lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        
        trainer_kwargs['callbacks'] = [cls.instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg if app_cfg_master['callback'].get(k, dict()).get('@off', False) == False]
        if app_cfg_master['callback']['bestCKPT'].get('@off', False) == False:
            trainer_kwargs['callbacks'].append(_checkpoint_callback)

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        for tl in trainer.loggers:
            handiCall = getattr(tl, 'setter_handiCall', lambda *args, **kwargs: None)
            handiCall(_trainer_obj=trainer)

        # configure learning rate ####################################
        bs, base_lr = config.data.params.batch_size, config.model.get('base_learning_rate', -1)
        if base_lr != -1:
            if not cpu:
                ngpu = len(lightning_config.trainer.gpus.strip(',').split(','))
            else:
                ngpu = 1
            accumulate_grad_batches = lightning_config.trainer.get('accumulate_grad_batches', 1)
            logger.info(f'accumulate_grad_batches = {accumulate_grad_batches}')
            lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            logger.info('Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)'.format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        
        puml('.config.puml', 'config.png', d=config)

        return model, trainer, data
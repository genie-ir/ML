# from omegaconf import OmegaConf

# a = OmegaConf.create()
# print(a, a.get('ali', 22))

# s = {
#     'name': 'mmd',
#     'cj': {'ali': 'alihejrati'},
#     'age': 32,
#     'fg': 44
# }

# a = s.pop('cj', dict())
# print(a)




# class A:
#     name = 'mmd'

#     def test(self):
#         print('!!!!!!!!!!!!', self.name)

# A().test()
        

# python main.py --base apps/VQGAN/configs/eyepacks_vqgan.yaml -t True --gpus 0,


# from omegaconf import OmegaConf

# a = OmegaConf.create({'name': 'mmd', 'age': 13})

# a = ''
# r = a or 2 or 21
# print(r)

# """
# Namespace(accelerator=None, accumulate_grad_batches=None, amp_backend='native', amp_level=None, auto_lr_find=False, auto_scale_batch_size=False, auto_select_gpus=False, benchmark=None, check_val_every_n_epoch=1, default_root_dir=None, detect_anomaly=False, deterministic=None, devices=None, enable_checkpointing=True, enable_model_summary=True, enable_progress_bar=True, fast_dev_run=False, gpus=None, gradient_clip_algorithm=None, gradient_clip_val=None, ipus=None, limit_predict_batches=None, limit_test_batches=None, limit_train_batches=None, limit_val_batches=None, log_every_n_steps=50, logger=True, max_epochs=None, max_steps=-1, max_time=None, min_epochs=None, min_steps=None, move_metrics_to_cpu=False, multiple_trainloader_mode='max_size_cycle', num_nodes=1, num_processes=None, num_sanity_val_steps=2, overfit_batches=0.0, plugins=None, precision=32, profiler=None, reload_dataloaders_every_n_epochs=0, replace_sampler_ddp=True, resume_from_checkpoint=None, strategy=None, sync_batchnorm=False, tpu_cores=None, track_grad_norm=-1, val_check_interval=None, weights_save_path=None)
# """

# import pytorch_lightning

# from pytorch_lightning.loggers import TensorBoardLogger

# print('!!', TensorBoardLogger)

# from omegaconf import OmegaConf
# a = OmegaConf.create({'name': 'mmd'})

# print(a.name)
# print(a.age) # error

# # print(a)
# print(a.mmd or 12)

# nowname = 22
# logdir = '/mmd/hooooooooooooooo!!'
# opt = {'debug': True}

# from libs.basicIO import readBIO
# cfg = readBIO('//apps/VQGAN/models/config.yaml')
# _st = str(cfg)
# for k, v in [('$nowname', nowname), ('$logdir', logdir), ('$opt.debug', opt['debug'])]:
#     _st = _st.replace(k, str(v))
# cfg = dict(_st)
# print(cfg)


# from pytorch_lightning.callbacks import ModelCheckpoint

# from test.shr import SignalHandler

# import signal

# trainer = {
#     'name': 'mmd',
#     'age': 13
# }

# print(trainer)
# SignalHandler(trainer)
# print(trainer)
# trainer['name'] = 'hooooooooooo!!'
# trainer['ok'] = 'nok'
# signal.pause()
# print(trainer)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# SignalHandler.melk()
# print(trainer)
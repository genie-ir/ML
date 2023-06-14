# print(isinstance(ARGS, ArgumentParser))
# print(bool(None))
# print(callable(lambda x: x+1))
# cfg = OmegaConf.create({'lightning': dict(**CONFIG)})
# print(cfg, type(cfg))
# print('!!!!!', CONFIG.logs.app.name, type(CONFIG))
# OmegaConf.save(dict(**CONFIG), pathBIO('//hoooooooooo2.yaml'))
# for s, m in zip([-3, 2, 21, 4, 232, 44, 23], [(6.12,3.21), (0.01,1.78), (6.3,1.5), (2.1,5.2), (12.36, 41.25), (54.58, 66.225), (22.456, 6.3)]):
#     metrics.add({
#         'FID': m[0] - 1.2,
#         'step': s,
#         'loss': m[0],
#         'val_loss': m[1]
#     })
# cfg = readBIO('/media/alihejrati/3E3009073008C83B/Code/Genie-ML/articles/taming-transformers-master/configs/imagenet_vqgan.yaml')
# t = 'articles.taming_transformers.taming.models.vqgan.VQModel' # cfg.model.target
# a, b = t.rsplit('.', 1)
# print(Import(t))
# print(CONFIG['logs']['data']['trainSet'])
# a = ls(CONFIG.logs.data.trainSet, '*.jpg')
# print(a)
# print(ARGS, getenv('NAME'))
# d, nc = 64, 300
# X = torch.randint(1,9, (8, d, 32, 32), dtype=torch.float32)
# C = torch.randint(1,9, (nc, d), dtype=torch.float32)
# X.requires_grad = True
# C.requires_grad = True
# Q, Qp, Xp = veqQuantizerImg(X, C)
# L = vqvae_loss(Xp, Q)
# print(L)
pass
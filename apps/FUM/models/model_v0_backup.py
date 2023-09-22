# DELETE this file shoude be delete later it is just for backuping of main code.


# self.train_ds, self.test_ds, self.val_ds, dataset_size = get_dataloader(cfg, 
#     # vqgan=self.vqgan,
#     tasknet=self.dr_classifire,
#     # drc=self.drc,
#     # vseg=self.vseg
# )
#
# ckpt = '/content/fine_tuned_weights/resnet50_128_08_100.pt'
# from torchvision import models
# weights = torch.load(ckpt)
# model = models.resnet50()
# # Our model outputs the score of DR for classification. See https://arxiv.org/pdf/2110.14160.pdf for more details.
# model.fc = nn.Linear(model.fc.in_features, 1)
# model.load_state_dict(weights, strict=True)
# self.drclassifire = model







# if phi_concept is not None:
        #     self.vqgan.save_phi((phi_concept), pathdir=self.pathdir, fname=f'0phi_concept.png')
        #     self.vqgan.save_phi((P0), pathdir=self.pathdir, fname=f'0phi_sprime.png')
        #     S = SSIM(phi_concept, P0, reduction='none').abs()
        #     ssim = (S>=.4).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).detach()
        #     print('S-------------->', S)
        #     print('ssim-------------->', ssim)
        #     P0 = (1-ssim) * P0 + ssim * phi_concept
        
        # P0 = (P0[:, 0:1, :,:] + P0[:, 1:2, :,:] + P0[:, 2:3, :,:]) / 3
        # P0 = torch.cat([P0, P0, P0], dim=1)





# def getdrmodel():
#     # call the model
#     model = vit_large_patch16(
#         num_classes=2,
#         drop_path_rate=0.2,
#         global_pool=True,
#     )

#     # load RETFound weights
#     ckpt = '/content/drive/MyDrive/storage/dependency/RETFound_cfp_weights.pth'
#     checkpoint = torch.load(ckpt, map_location='cpu')
#     checkpoint_model = checkpoint['model']
#     state_dict = model.state_dict()
#     for k in ['head.weight', 'head.bias']:
#         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#             print(f"Removing key {k} from pretrained checkpoint")
#             del checkpoint_model[k]

#     # interpolate position embedding
#     interpolate_pos_embed(model, checkpoint_model)

#     # load pre-trained model
#     msg = model.load_state_dict(checkpoint_model, strict=False)

#     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

#     # manually initialize fc layer
#     trunc_normal_(model.head.weight, std=2e-5)

#     # print("Model = %s" % str(model))
#     return model


    # def training_step2(self, batch, batch_idx, split='train'):
    #     _std = torch.tensor([0.1252, 0.0857, 0.0814], device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    #     _mean = torch.tensor([0.3771, 0.2320, 0.1395], device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    #     phi = self.vqgan.lat2phi(batch['X'].float().flatten(1))
    #     _phi = self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'batch.png', sreturn=True).to(self.device)
        
    #     from einops import rearrange
    #     import torchvision, numpy as np
    #     img = torchvision.utils.make_grid(
    #         _phi.detach().cpu(), 
    #         nrow=2
    #     )
    #     img = rearrange(img, 'c h w -> h w c').contiguous()
    #     # img = rearrange(img, 'b c h w -> b h w c').contiguous()
    #     img = img.numpy().astype(np.uint8)
    #     signal_save(img, self.pathdir + '/' + f'_batch.png')


    #     _phi = F.interpolate(_phi, size=(512, 512), mode='bilinear', align_corners=False)
    #     img = torchvision.utils.make_grid(
    #         _phi.detach().cpu(), 
    #         nrow=2
    #     )
    #     img = rearrange(img, 'c h w -> h w c').contiguous()
    #     # img = rearrange(img, 'b c h w -> b h w c').contiguous()
    #     img = img.numpy().astype(np.uint8)
    #     signal_save(img, self.pathdir + '/' + f'_batch2.png')
        


    #     _phi = (_phi - _mean * 255) / (_std * 255)
    #     print(self.drclassifire(_phi))
    #     print(batch['y'])
    #     assert False







# for i in range(5):
        #     for j in range(5):
        #         print(f'SSIM(phi{i}, phi{j})=', SSIM(phi[i:i+1], phi[j:j+1]).abs())
        # print(f'--> SSIM(phi, phi)=', SSIM(phi, phi).abs())
        # print(f'--> SSIM(phi, permute(phi))=', SSIM(phi, torch.cat([
        #     phi[2:3], phi[0:1], phi[4:5], phi[1:2], phi[3:4]
        # ], dim=0)).abs())



class _FUM(plModuleBase):
    def resnet50(self, model):
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    def validation_step(self, batch, batch_idx, split='val'):
        pass
    
    def __start(self):
        self.seqlen = 3 # 256
        self.seqdim = 2 # 1
        self.vocab_size = 451
        self.transformer = Transformer(
            heads=getattr(self, 'heads', 1),
            maxlen=getattr(self, 'maxlen', 1e3),
            dropout=getattr(self, 'dropout', 0),
            fwd_expan=getattr(self, 'fwd_expan', 4),
            num_layers=getattr(self, 'num_layers', 8),
            trg_mask=getattr(self, 'trg_mask', True),
            src_mask=getattr(self, 'src_mask', False),
            embed_size=getattr(self, 'latent_dim', 256),
            trg_vocab_size=getattr(self, 'trg_vocab_size', 1e3),
            src_vocab_size=getattr(self, 'src_vocab_size', 1e3)
        )

    def generator_step(self, batch):
        # latent = self.ccodebook(batch[self.signal_key])[0].view(-1, self.qwh, self.qwh)
        latent = batch[self.signal_key].float()
        old_rec_metric = -1
        phi_shape = (batch['batch_size'], self.phi_ch, self.phi_wh, self.phi_wh)
        s1 = torch.zeros(phi_shape, device=self.device)
        # s2 = torch.zeros(phi_shape, device=self.device)
        for N in range(1, self.phi_steps + 1):
            phi = self.vqgan.lat2phi(latent)
            s1 = s1 + phi
            break
            # s2 = s2 + phi ** 2
            latent_rec = self.vqgan.phi2lat(phi).float()
            rec_metric = (latent-latent_rec).abs().sum()
            # print('--lm-->', rec_metric)
            latent = latent_rec
            # self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'phi-{str(N)}.png')
            if rec_metric < 1e-6 or old_rec_metric == rec_metric:
                break
            old_rec_metric = rec_metric
        # compressor(self.pathdir, self.pathdir + '/phi.zip')
        phi = s1 / N
        # print('!!!!!!!!!!!!! mue', mue.shape, mue.dtype, mue.requires_grad)
        p = self.vqgan.phi2lat(phi).float().flatten(1).unsqueeze(-1).unsqueeze(-1)
        # print('!!!!!!!!!!!!!! p', p.shape, p.dtype, p.requires_grad)
        s, sloss = self.scodebook(p)
        # print('!!!!!!!!!!!!!! s', s.shape, s.dtype, s.requires_grad)
        # sq = self.qw * self.vqgan.lat2qua(s) + self.qb
        # sq = self.vqgan.lat2qua(s)
        # print('++++++++++++++>', batch['y'])
        self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'final/phi-{str(N)}.png')
        
        dloss_phi = -torch.mean(self.vqgan.loss.discriminator(phi))
        loss_phi = self.LeakyReLU(dloss_phi - self.gamma)
        loss = self.lambda_loss_phi * loss_phi 
        ld = dict()
        for c in range(self.nclasses):
            scphic = self.vqgan.qua2phi(self.mac[c](self.vqgan.lat2qua(s)))
            print('-------------->', c)
            # print('-------------->', self.drclassifire(scphic))
            self.vqgan.save_phi(scphic, pathdir=self.pathdir, fname=f'final/scphic({c})-{str(N)}.png')
            dloss_scphic = -torch.mean(self.vqgan.loss.discriminator(scphic))
            loss_scphic = self.lambda_loss_scphic[c] * self.LeakyReLU(dloss_scphic - self.gamma)
            drloss_scphic = self.lambda_drloss_scphic[c] * torch.tensor(1, device=self.device) #* self.drclassifire(scphic).mean()
            ld[f'loss_scphic_{c}'] = loss_scphic.clone().detach().mean()
            ld[f'drloss_scphic_{c}'] = drloss_scphic.clone().detach().mean()
            loss = loss + loss_scphic + drloss_scphic

        lossdict = self.generatorLoss._lossdict(
            loss=loss,
            loss_phi=loss_phi,
            dloss_phi=dloss_phi,
            **ld
        )

        print('@@@@@@@@@@@@@@@', lossdict)

        assert False
        return loss, lossdict
    
    def start(self):
        if not isinstance(self.lambda_loss_scphic, (list, tuple)):
            lambda_loss_scphic = float(self.lambda_loss_scphic)
            self.lambda_loss_scphic = [lambda_loss_scphic for c in range(self.nclasses)]
        
        if not isinstance(self.lambda_drloss_scphic, (list, tuple)):
            lambda_drloss_scphic = float(self.lambda_drloss_scphic)
            self.lambda_drloss_scphic = [lambda_drloss_scphic for c in range(self.nclasses)]

        self.qshape = (self.qch, self.qwh, self.qwh)
        self.mac = nn.Sequential(*[
            MAC(units=2, shape=self.qshape) for c in range(self.nclasses)
        ])
        # self.qw = nn.Parameter(torch.randn(self.qshape))
        # self.qb = nn.Parameter(torch.randn(self.qshape))
        self.scodebook = VectorQuantizer(n_e=self.ncluster, e_dim=self.latent_dim, beta=0.25, zwh=1)
        self.ccodebook = VectorQuantizer(n_e=(self.ncrosses * self.ncluster), e_dim=self.latent_dim, beta=0.25, zwh=1)
        self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)

    # def generator_step00(self, batch):
    #     x = self.codebook(batch[self.signal_key])
    #     phi = self.vqgan.rec_phi({'x': x, 'y': batch['y']})
    #     self.vqgan.save_phi(phi, pathdir='/content')

    #     g_loss = -torch.mean(self.vqgan.loss.discriminator(phi.contiguous()))
    #     print('g_loss', g_loss.shape, g_loss, g_loss.requires_grad)
    #     assert False
    #     return g_loss, {'loss': g_loss.item()}

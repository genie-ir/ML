import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor
from utils.pt.BB.Calculation.residual_block import MAC
from utils.pt.BB.Scratch.Transformer.transformer import Transformer
from utils.pt.BB.Quantizer.VectorQuantizer import VectorQuantizer2 as VectorQuantizer

class FUM(plModuleBase):
    def validation_step(self, batch, batch_idx, split='val'):
        pass
    
    # def on_train_epoch_end(self):
    #     assert False

    def training_step(self, batch, batch_idx, split='train'):
        # B = batch[self.signal_key]
        B = torch.tensor([2,2,2,2, 6], device=self.device)
        b = B[0]
        self.b = b
        print('1111111111111111111', b)
        print(f'iter{batch_idx} | before', self.generator.ccodebook.embedding.weight[b,0])
        for C in range(self.nclasses):
            batch['C'] = C
            batch[self.signal_key] = self.generator.ccodebook.fwd_nbpi(B) #.clone()
            print(f'B{batch_idx}', batch[self.signal_key].shape, batch[self.signal_key].dtype, batch[self.signal_key].requires_grad)
            batch[self.signal_key].requires_grad_(True)
            super().training_step(batch, batch_idx, split)
        print(f'iter{batch_idx} | after', self.generator.ccodebook.embedding.weight[b,0])
        if batch_idx == 2:
            assert False, batch_idx
    
    def resnet50(self, model):
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    # def configure_optimizers(self):
    #     print('#############################################')
    #     return super().configure_optimizers()
    
    def start(self):
        self.hp('lambda_loss_scphi', (list, tuple), len=self.nclasses)
        self.hp('lambda_drloss_scphi', (list, tuple), len=self.nclasses)
        self.qshape = (self.qch, self.qwh, self.qwh)
        self.phi_shape = (self.phi_ch, self.phi_wh, self.phi_wh)
        self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)
        self.generator.scodebook = VectorQuantizer(ncluster=self.ncluster, dim=self.latent_dim, zwh=1)
        self.generator.ccodebook = VectorQuantizer(ncluster=(self.ncrosses * self.ncluster), dim=self.latent_dim, zwh=1)
        self.generator.mac = nn.Sequential(*[
            MAC(units=2, shape=self.qshape) for c in range(self.nclasses)
        ])

    def __c2phi(self, c, batch_size):
        latent = c
        old_rec_metric = -1
        s1 = torch.zeros((batch_size,) + self.phi_shape, device=self.device)
        # s2 = torch.zeros(phi_shape, device=self.device)
        for N in range(1, self.phi_steps + 1):
            phi = self.vqgan.lat2phi(latent)
            self.sethooks(latent, hooks=lambda grad: print('@@@@@@@@@@@', grad.shape, grad[0, :3], grad[-1, :3]))
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
        return s1 / N
    
    def generator_step(self, batch):
        C = batch['C']
        print('!!!!!!!!!!!', C)
        c = batch[self.signal_key] # dataset -> replace -> selection of ccodebook
        phi = self.__c2phi(c, batch['batch_size'])
        p = self.vqgan.phi2lat(phi).float().flatten(1).unsqueeze(-1).unsqueeze(-1) #NOTE derivative?
        s, sloss = self.generator.scodebook(p)
        sq = self.vqgan.lat2qua(s)
        scphi = self.vqgan.qua2phi(self.generator.mac[C](sq))

        dloss_phi = -torch.mean(self.vqgan.loss.discriminator(phi))
        dloss_scphi = -torch.mean(self.vqgan.loss.discriminator(scphi))
        loss_phi = self.lambda_loss_phi * self.LeakyReLU(dloss_phi - self.gamma)
        loss_scphi = self.lambda_loss_scphi[C] * self.LeakyReLU(dloss_scphi - self.gamma)
        drloss_scphi = self.lambda_drloss_scphi[C] * torch.ones((1,), device=self.device) #* self.drclassifire(scphic).mean()
        loss = loss_phi + loss_scphi + drloss_scphi
        
        lossdict = self.generatorLoss.lossdict(
            loss=loss,
            loss_phi=loss_phi,
            dloss_phi=dloss_phi,
            loss_scphi=loss_scphi,
            dloss_scphi=dloss_scphi,
            drloss_scphi=drloss_scphi,
            Class=torch.tensor(float(C))
        )

        print('@@@@@@@@@@@@@@@', lossdict)

        # self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'final/Class-{C}/phi.png')
        # self.vqgan.save_phi(scphi, pathdir=self.pathdir, fname=f'final/Class-{C}/scphi.png')

        return loss, lossdict





























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

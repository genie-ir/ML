import torch
from torch import nn
from libs.basicIO import dfdir
from utils.pl.plModuleBase import plModuleBase
from utils.pt.BB.Calculation.residual_block import MAC
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
from utils.pt.BB.Quantizer.VectorQuantizer import VectorQuantizer as VectorQuantizerBase


# TODO we looking for uniqness.
class VectorQuantizer(VectorQuantizerBase):
    def embedding_weight_init(self):
        dataset_path = '/content/drive/MyDrive/storage/pretrained_0_1DsignalOfEyepacs.zip'
        dfdir(src_file=dataset_path, src_dir='/content/dataset')
        assert False
        t = torch.randint(0, 1024, self.eshape) # NOTE dataset
        # t = torch.randint(0, 1024, self.eshape) # NOTE random latents
        self.w = self.nnParameter(tensor=t.log())
        self.embedding.weight = self.w

class FUM(plModuleBase):
    def validation_step(self, batch, batch_idx, split='val'):
        return
    
    def training_step(self, batch, batch_idx, split='train'):
        if batch_idx == 0:
            print('-'*60)
            print(self.generator.scodebook.embedding.weight)
            print('-'*60)

        # B = batch[self.signal_key]
        t = 3612
        B = torch.tensor([2,2,2,2, t], device=self.device)
        b = B[0]
        print(f'iter{batch_idx} | before', self.generator.scodebook.embedding.weight[b,0], self.generator.scodebook.embedding.weight[b,0].exp())
        for cidx in range(self.nclasses):
            # print('----grad---->', self.generator.scodebook.embedding.weight.grad)
            batch['cidx'] = cidx
            batch['bidx'] = batch_idx
            batch[self.signal_key] = self.generator.scodebook.fwd_nbpi(B).exp() #.clone()
            super().training_step(batch, batch_idx, split)
        print(f'iter{batch_idx} | after', self.generator.scodebook.embedding.weight[b,0], self.generator.scodebook.embedding.weight[b,0].exp())
        if batch_idx == 2:
            assert False, batch_idx
    
    def resnet50(self, model):
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    def start(self):
        self.hp('lambda_loss_scphi', (list, tuple), len=self.nclasses)
        self.hp('lambda_drloss_scphi', (list, tuple), len=self.nclasses)
        self.qshape = (self.qch, self.qwh, self.qwh)
        self.phi_shape = (self.phi_ch, self.phi_wh, self.phi_wh)
        self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)
        self.generator.scodebook = VectorQuantizer(ncluster=self.ncluster, dim=self.latent_dim, zwh=1)
        # self.generator.ccodebook = VectorQuantizer(ncluster=(self.ncrosses * self.ncluster), dim=self.latent_dim, zwh=1)
        self.generator.mac = nn.Sequential(*[
            MAC(units=2, shape=self.qshape) for c in range(self.nclasses)
        ])

    def __c2phi(self, cross, batch_size):
        # list_of_distance_to_mode = []
        # BASIC the very basic code of idea behind chaining concept.
        # phi = self.vqgan.lat2phi(cross)
        # sn = self.vqgan.phi2lat(phi).flatten(1).float().detach()
        # return phi, sn
        
        # IDEA s1 = phi0 # is better than --> torch.zeros((batch_size,) + self.phi_shape, device=self.device) --> becuse the first one is diffrentiable. NOTE: each time you must do: s1=s1+phi
        phi0 = self.vqgan.lat2phi(cross)
        nl = self.vqgan.phi2lat(phi0.detach()).float()
        # _np = phi0.detach()
        for N in range(1, self.phi_steps):
            # list_of_distance_to_mode.append(nl.flatten(1))
            np = self.vqgan.lat2phi(nl)
            # print(f'({N-1},{N})-------ssim-------->', SSIM(_np, np))
            nnl = self.vqgan.phi2lat(np).float()
            qe_mse = ((nl-nnl)**2).mean()
            nl = nnl
            if qe_mse < 1e-6: 
                break
            # _np = np
            # print(f'{N} ---qe_mse-->', qe_mse.item())
            self.vqgan.save_phi(np, pathdir=self.pathdir, fname=f'phi-{str(N)}.png')
        # mue = (s1 / N).detach()
        sn = nl.flatten(1).detach()
        # print('='*60)
        # for i, l in enumerate(list_of_distance_to_mode):
        #     print(f'{i}--->', ((l-sn)**2).mean().item())
        return phi0, sn
    
    def generator_step(self, batch):
        bidx = batch['bidx']
        cidx = batch['cidx']
        ln = batch[self.signal_key]
        phi, sn = self.__c2phi(ln, batch['batch_size'])
        # SN = self.generator.scodebook.fwd_nbpi(self.generator.scodebook.fwd_getIndices(sn.unsqueeze(-1).unsqueeze(-1)).squeeze())
        
        # s, sloss = self.generator.scodebook(p)
        # sq = self.vqgan.lat2qua(s)
        # scphi = self.vqgan.qua2phi(self.generator.mac[C](sq))

        dloss_phi = -torch.mean(self.vqgan.loss.discriminator(phi)) # NOTE DLOSS.shape=(B,1,30,30) float32.
        loss_phi = self.lambda_loss_phi * self.LeakyReLU(dloss_phi - self.gamma)
        loss_latent = self.lambda_loss_latent * self.generatorLoss.lossfn_p1log(ln, sn)
        # dloss_scphi = -torch.mean(self.vqgan.loss.discriminator(scphi))
        # loss_scphi = self.lambda_loss_scphi[C] * self.LeakyReLU(dloss_scphi - self.gamma)
        # drloss_scphi = self.lambda_drloss_scphi[C] * torch.ones((1,), device=self.device) #* self.drclassifire(scphic).mean()
        loss = loss_latent + loss_phi #+ loss_scphi + drloss_scphi
        
        lossdict = self.generatorLoss.lossdict(
            loss=loss,
            loss_phi=loss_phi,
            dloss_phi=dloss_phi,
            loss_latent=loss_latent,
            # loss_scphi=loss_scphi,
            # dloss_scphi=dloss_scphi,
            # drloss_scphi=drloss_scphi,
            Class=torch.tensor(float(cidx))
        )
        print(f'cidx={cidx}', lossdict)

        # self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/phi.png')
        # self.vqgan.save_phi(scphi, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/scphi.png')

        return loss, lossdict

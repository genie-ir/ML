import torch
from torch import nn
from libs.basicIO import dfdir
from utils.pt.tricks.gradfns import dzq_dz_eq1
from utils.pl.plModuleBase import plModuleBase
from utils.pt.BB.Calculation.residual_block import MAC
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
from utils.pt.BB.Quantizer.VectorQuantizer import VectorQuantizer as VectorQuantizerBase


# TODO we looking for uniqness.
class VectorQuantizer(VectorQuantizerBase):
    def embedding_weight_init(self):
        # dataset_path = '/content/drive/MyDrive/storage/pretrained_0_1DsignalOfEyepacs.zip'
        # dfdir(src_file=dataset_path, src_dir='/content/dataset')
        # assert False
        # t = torch.randint(0, 1024, self.eshape) # NOTE dataset
        t = torch.randint(0, 1024, self.eshape) # NOTE random latents
        # t = torch.randint(0, 10, self.eshape) # NOTE random latents
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
        B = torch.tensor([2,3,4,5, t], device=self.device)
        # B = torch.tensor([2], device=self.device)
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

    def __c2phi(self, cross, tag='', phi_concept=None):
        # list_of_distance_to_mode = []
        # BASIC the very basic code of idea behind chaining concept.
        # phi = self.vqgan.lat2phi(cross)
        # sn = self.vqgan.phi2lat(phi).flatten(1).float().detach()
        # return phi, sn
        
        # IDEA s1 = phi0 # is better than --> torch.zeros((batch_size,) + self.phi_shape, device=self.device) --> becuse the first one is diffrentiable. NOTE: each time you must do: s1=s1+phi
        phi0 = self.vqgan.lat2phi(cross)
        P0 = phi0.clone().detach()
        if phi_concept is not None:
            self.vqgan.save_phi((phi_concept), pathdir=self.pathdir, fname=f'0phi_concept.png')
            self.vqgan.save_phi((P0), pathdir=self.pathdir, fname=f'0phi_sprime.png')
            S = SSIM(phi_concept, P0, reduction='none').abs()
            ssim = (S>=.4).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).detach()
            print('S-------------->', S)
            print('ssim-------------->', ssim)
            P0 = (1-ssim) * P0 + ssim * phi_concept
        
        # P0 = (P0[:, 0:1, :,:] + P0[:, 1:2, :,:] + P0[:, 2:3, :,:]) / 3
        # P0 = torch.cat([P0, P0, P0], dim=1)
        nl = self.vqgan.phi2lat(P0).float()
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
            self.vqgan.save_phi(np, pathdir=self.pathdir, fname=f'{tag}phi-{str(N)}.png')
        # mue = (s1 / N).detach()
        sn = nl.flatten(1).detach()
        np = np.detach()
        # print('='*60)
        # for i, l in enumerate(list_of_distance_to_mode):
        #     print(f'{i}--->', ((l-sn)**2).mean().item())
        return phi0, sn, np
    
    def generator_step(self, batch):
        cidx = batch['cidx']
        ln = batch[self.signal_key]
        phi, sn, concept = self.__c2phi(ln) # NOTE `sn` and `concept` doesnt have derevetive.
        
        nidx = self.generator.scodebook.fwd_getIndices(sn.unsqueeze(-1).unsqueeze(-1)).squeeze()
        s_prime = self.generator.scodebook.fwd_nbpi(nidx).exp().detach()
        
        
        phi_sprime, s_zegond, phi_szegond = self.__c2phi(s_prime, phi_concept=phi.detach()) # NOTE `phi_sprime` does not exist in any deravitive path.
        
        sn = dzq_dz_eq1(sn, ln)
        divergenceloss = 1 - torch.sigmoid((1 + ((sn-s_zegond) ** 2).mean()).log())

        # s, sloss = self.generator.scodebook(p)
        # sq = self.vqgan.lat2qua(s)
        # scphi = self.vqgan.qua2phi(self.generator.mac[C](sq))

        dloss_phi = -torch.mean(self.vqgan.loss.discriminator(phi)) # NOTE DLOSS.shape=(B,1,30,30) float32.
        loss_phi = self.lambda_loss_phi * self.LeakyReLU(dloss_phi - self.gamma)
        convergenceloss = self.lambda_loss_latent * self.generatorLoss.lossfn_p1log(ln, sn)
        # dloss_scphi = -torch.mean(self.vqgan.loss.discriminator(scphi))
        # loss_scphi = self.lambda_loss_scphi[C] * self.LeakyReLU(dloss_scphi - self.gamma)
        # drloss_scphi = self.lambda_drloss_scphi[C] * torch.ones((1,), device=self.device) #* self.drclassifire(scphic).mean()
        
        
        self.vqgan.save_phi(concept, pathdir=self.pathdir, fname=f'concept.png')
        self.vqgan.save_phi(phi_sprime, pathdir=self.pathdir, fname=f'phi_sprime.png')
        self.vqgan.save_phi(phi_szegond, pathdir=self.pathdir, fname=f'phi_szegond.png')
        
        print('----nidx------->', nidx)
        
        loss = convergenceloss + divergenceloss + loss_phi #+ loss_scphi + drloss_scphi
        
        lossdict = self.generatorLoss.lossdict(
            loss=loss,
            loss_phi=loss_phi,
            dloss_phi=dloss_phi,
            divergenceloss=divergenceloss,
            convergenceloss=convergenceloss,
            # loss_scphi=loss_scphi,
            # dloss_scphi=dloss_scphi,
            # drloss_scphi=drloss_scphi,
            Class=torch.tensor(float(cidx))
        )
        print(f'cidx={cidx}', lossdict)

        # self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/phi.png')
        # self.vqgan.save_phi(scphi, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/scphi.png')

        return loss, lossdict

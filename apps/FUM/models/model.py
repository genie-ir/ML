import torch
from torch import nn
from utils.pl.plModuleBase import plModuleBase
from utils.pt.BB.Calculation.residual_block import MAC
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
from utils.pt.BB.Quantizer.VectorQuantizer import VectorQuantizer as VectorQuantizerBase

# TODO we looking for uniqness.
class VectorQuantizer(VectorQuantizerBase):
    def embedding_weight_init(self):
        self.w = self.nnParameter(tensor=torch.randint(0, 1024, self.eshape).log())
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
            print('----grad---->', self.generator.scodebook.embedding.weight.grad)
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
        # BASIC the very basic code of idea behind chaining concept.
        # phi = self.vqgan.lat2phi(cross)
        # sn = self.vqgan.phi2lat(phi).flatten(1).float().detach()
        # return phi, sn
        
        old_quantization_error = float('inf')
        phi0 = self.vqgan.lat2phi(cross)
        # latent = self.vqgan.phi2lat(phi0).float().detach()
        phi = phi0.detach()
        # IDEA s1 = phi0 # is better than --> torch.zeros((batch_size,) + self.phi_shape, device=self.device)   becuse the first one is diffrentiable.
        for N in range(1, self.phi_steps):
            # s1 = s1 + phi
            latent_index = self.vqgan.phi2lat(phi).float()
            phi_of_latent_index = self.vqgan.lat2phi(latent_index)
            quantization_error = ((phi-phi_of_latent_index).abs()).sum()
            latent = latent_index
            phi = phi_of_latent_index
            self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'phi-{str(N)}.png')
            # print('latent', latent.shape, latent.dtype)
            # print('latent_index', latent_index.shape, latent_index.dtype)
            print('---quantization_error-->', quantization_error.item())
            print(f'{N}--- old_quantization_error - quantization_error --->', (old_quantization_error - quantization_error).item(), (old_quantization_error - quantization_error).item() < 1e-6)
            if quantization_error < 1e-6 or (old_quantization_error - quantization_error) < 1e-6:
                break
            old_quantization_error = quantization_error
            
        # compressor(self.pathdir, self.pathdir + '/phi.zip')
        # mue = (s1 / N).detach()
        sn = latent.detach()
        assert False
        return phi0, sn
    
    def generator_step(self, batch):
        bidx = batch['bidx']
        cidx = batch['cidx']
        ln = batch[self.signal_key]
        mue, sn = self.__c2phi(ln.detach(), batch['batch_size'])
        SN = self.generator.scodebook.fwd_getIndices(sn.unsqueeze(-1).unsqueeze(-1)).squeeze()
        # print('ln', ln.shape, ln.dtype)
        # print('phi', phi.shape, phi.dtype)
        # print('sn', sn.shape, sn.dtype)
        # print('SN', SN.shape, SN.dtype, SN) #NOTE: index of nearset latents to sn
        
        
        
        
        
        
        # s, sloss = self.generator.scodebook(p)
        # sq = self.vqgan.lat2qua(s)
        # scphi = self.vqgan.qua2phi(self.generator.mac[C](sq))

        
        DLOSS = self.vqgan.loss.discriminator(phi)
        print('DLOSS', DLOSS.shape, DLOSS.dtype)
        dloss_phi = -torch.mean(DLOSS)
        loss_latent = self.lambda_loss_latent * self.generatorLoss.lossfn_p1log(ln, sn)
        # dloss_scphi = -torch.mean(self.vqgan.loss.discriminator(scphi))
        loss_phi = self.lambda_loss_phi * self.LeakyReLU(dloss_phi - self.gamma)
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
        # for i in range(5):
        #     for j in range(5):
        #         print(f'SSIM(phi{i}, phi{j})=', SSIM(phi[i:i+1], phi[j:j+1]).abs())
        # print(f'--> SSIM(phi, phi)=', SSIM(phi, phi).abs())
        # print(f'--> SSIM(phi, permute(phi))=', SSIM(phi, torch.cat([
        #     phi[2:3], phi[0:1], phi[4:5], phi[1:2], phi[3:4]
        # ], dim=0)).abs())

        # self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/phi.png')
        # self.vqgan.save_phi(scphi, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/scphi.png')

        return loss, lossdict

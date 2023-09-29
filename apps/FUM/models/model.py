try:
    import torch
    from torch import nn
    from libs.basicIO import dfdir
    import torch.nn.functional as F
    from libs.basicIO import signal_save
    from utils.pt.tricks.gradfns import dzq_dz_eq1
    from utils.pl.plModuleBase import plModuleBase
    from utils.pt.BB.Calculation.residual_block import MAC
    from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
    from utils.pt.BB.Quantizer.VectorQuantizer import VectorQuantizer as VectorQuantizerBase
    from torch.utils.data import DataLoader


    # import tensorflow as tf
    # from tensorflow.keras.applications.inception_v3 import InceptionV3
    # from tensorflow.keras.models import Model
    # from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, Flatten

    from dependency.MKCNet.main import pretrain as makeDRclassifire, get_transform
except Exception as e:
    assert False, e

try:
    import tensorflow.keras
    import torch
    from dependency.RETFound_MAE.models_vit import vit_large_patch16
    from dependency.RETFound_MAE.util.pos_embed import interpolate_pos_embed
    from timm.models.layers import trunc_normal_
except Exception as e:
    print(e)

from libs.basicIO import cmatrix
from utils.pt.tricks.gradfns import onehot_with_grad, dzq_dz_eq1


from libs.coding import random_string


from dependency.MKCNet.dataset.dataset_manager import get_dataloader
from dependency.BCDU_Net.Retina_Blood_Vessel_Segmentation.pretrain import pretrain as makevaslsegmentation


from utils.plots.plot1d import Plot1D


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
    # def train_dataloader(self):
    #     return DataLoader(self.train_ds, batch_size=5,shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_ds, batch_size=5,shuffle=False)

    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size=5,shuffle=False)

    # def resnet50(self, model):
    #     model.fc = nn.Linear(model.fc.in_features, 1)
    #     return model
    
    
    # NOTE: DR_CLASSIFIRE Training function.
    def generator_step__drcalgo(self, batch, **kwargs):
        phi = self.vqgan.lat2phi(batch['X'].flatten(1).float())
        phi_denormalized = self.vqgan_fn_phi_denormalize(phi).detach()
        # signal_save(phi_denormalized, f'/content/a.png', stype='img', sparams={'chw2hwc': True})
        phi_denormalized = (phi_denormalized - (self.dr_classifire_normalize_mean * 255)) / (self.dr_classifire_normalize_std * 255)
        output, output_M, output_IQ = self.generator.dr_classifire(phi_denormalized)
        dr_pred = self.generator.softmax(output)
        loss = self.generator.ce(dr_pred, batch['y_edit'])
        
        # print('11111111111111', dr_pred)
        # print('groundtrouth -> y_edit', batch['y_edit'])
        # print('------------------------->', loss)
        
        # if kwargs['split'] == 'train':
        #     self.t_ypred = self.t_ypred + list(dr_pred.argmax(dim=1).cpu().numpy())
        #     self.t_ygrnt = self.t_ygrnt + list(batch['y_edit'].cpu().numpy())
        # else:
        #     self.v_ypred = self.v_ypred + list(dr_pred.argmax(dim=1).cpu().numpy())
        #     self.v_ygrnt = self.v_ygrnt + list(batch['y_edit'].cpu().numpy())
        return loss, dict(loss=loss.cpu().detach().item())

    
    # NOTE: Synthesis Algorithm.
    def training_step__synalgo(self, batch, batch_idx, split='train'):
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
    
    def start(self, dr_vs_synthesis_flag=True):
        self.dr_classifire_normalize_std = torch.tensor([0.1252, 0.0857, 0.0814]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to('cuda')
        self.dr_classifire_normalize_mean = torch.tensor([0.3771, 0.2320, 0.1395]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to('cuda')

        self.vqgan_fn_phi_denormalize = lambda G: ((((G.clamp(-1., 1.))+1)/2)*255)#.transpose(0,1).transpose(1,2)
        self.generator.ce = nn.CrossEntropyLoss()
        self.generator.softmax = torch.nn.Softmax(dim=1)
        self.t_ypred = []
        self.t_ygrnt = []
        self.v_ypred = []
        self.v_ygrnt = []
        
        # self.vseg = makevaslsegmentation('/content/drive/MyDrive/storage/dr_classifire/unet-segmentation/weight_retina.hdf5')
        # self.vseg.requires_grad_(False)

        self.dr_classifire, cfg = makeDRclassifire('/content/drive/MyDrive/storage/dr_classifire/best_model.pth')
        self.dr_classifire = self.dr_classifire.to('cuda')
        
        if dr_vs_synthesis_flag:
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
            self.dr_classifire.requires_grad_(False)
        else:
            self.generator.dr_classifire = self.dr_classifire

    def __c2phi(self, cross, tag='', phi_concept=None):
        # list_of_distance_to_mode = []
        # BASIC the very basic code of idea behind chaining concept.
        # phi = self.vqgan.lat2phi(cross)
        # sn = self.vqgan.phi2lat(phi).flatten(1).float().detach()
        # return phi, sn

        PHI = []
        PHI_L = []
        
        # IDEA s1 = phi0 # is better than --> torch.zeros((batch_size,) + self.phi_shape, device=self.device) --> becuse the first one is diffrentiable. NOTE: each time you must do: s1=s1+phi
        Q = self.vqgan.lat2qua(cross)
        phi0 = self.vqgan.qua2phi(Q)
        nl = self.vqgan.phi2lat(phi0.detach()).float()
        # _np = phi0.detach()
        for N in range(1, self.phi_steps):
            # list_of_distance_to_mode.append(nl.flatten(1))
            np = self.vqgan.lat2phi(nl)


            PHI.append(np[0].cpu().detach().numpy())
            
            
            # print(f'({N-1},{N})-------ssim-------->', SSIM(_np, np))
            nnl = self.vqgan.phi2lat(np).float()
            # qe_mse = ((nl-nnl)**2).mean()
            qe_mse = ((nl-nnl).abs()).sum()

            PHI_L.append(qe_mse[0].item())

            nl = nnl
            if qe_mse < 1e-6: 
                break
            # _np = np
            # print(f'{N} ---qe_mse-->', qe_mse.item())
            # self.vqgan.save_phi(np, pathdir=self.pathdir, fname=f'{tag}phi-{str(N)}.png')
        # mue = (s1 / N).detach()
        sn = nl.flatten(1).detach()
        np = np.detach()
        # print('='*60)
        # for i, l in enumerate(list_of_distance_to_mode):
        #     print(f'{i}--->', ((l-sn)**2).mean().item())
        
        
        print('QQQQQQQQQQQQQQQQQQQQQQQQQQ', PHI_L.shape)
        self.vqgan.save_phi(torch.cat(PHI, dim=0), pathdir=self.pathdir, fname=f'PHI.png')
        neon = Plot1D(xlabel='Iteration', ylabel='Quantization Error')
        neon.plot(range(len(PHI_L)), PHI_L, label=f'convergence curve')
        neon.savefig('/content/convergence.png')
        assert False
        return (phi0, Q), sn, np
    
    def generator_step__synalgo(self, batch, **kwargs):
        bidx = batch['bidx']
        cidx = batch['cidx']
        ln = batch[self.signal_key]

        # print(ln.shape)
        # neon = Plot1D(xlabel='latent index', ylabel='codebook index')
        # neon.plot(range(256), ln[0].cpu().detach().numpy(), label=f'Latent Code')
        # neon.savefig('/content/plot.png')
        # assert False


        (phi, q_phi), sn, concept = self.__c2phi(ln) # NOTE `sn` and `concept` doesnt have derevetive.
        
        cphi = self.vqgan.qua2phi(self.generator.mac[cidx](q_phi))
        cphi_denormalized = self.vqgan_fn_phi_denormalize(cphi).detach()
        cphi_denormalized = dzq_dz_eq1(cphi_denormalized, cphi)
        cphi_denormalized = (cphi_denormalized - (self.dr_classifire_normalize_mean * 255)) / (self.dr_classifire_normalize_std * 255)
        output_DR, output_M, output_IQ = self.dr_classifire(cphi_denormalized)
        dr_pred = self.generator.softmax(output_DR)
        drloss = self.lambda_drloss_scphi[cidx] * self.generator.ce(dr_pred, batch['y_edit'])

        nidx = self.generator.scodebook.fwd_getIndices(sn.unsqueeze(-1).unsqueeze(-1)).squeeze()
        s_prime = self.generator.scodebook.fwd_nbpi(nidx).exp().detach()
        (phi_sprime, phisprime_q), s_zegond, phi_szegond = self.__c2phi(s_prime, phi_concept=phi.detach()) # NOTE `phi_sprime` does not exist in any deravitive path.
        
        sn = dzq_dz_eq1(sn, ln)
        divergenceloss = 1 - torch.sigmoid((1 + ((sn-s_zegond) ** 2).mean()).log())
        convergenceloss = self.lambda_loss_latent * self.generatorLoss.lossfn_p1log(ln, sn.detach())

        dloss_phi = -torch.mean(self.vqgan.loss.discriminator(phi)) # NOTE DLOSS.shape=(B,1,30,30) float32.
        loss_phi = self.lambda_loss_phi * self.LeakyReLU(dloss_phi - self.gamma)
        dloss_cphi = -torch.mean(self.vqgan.loss.discriminator(cphi)) # NOTE DLOSS.shape=(B,1,30,30) float32.
        loss_cphi = self.lambda_loss_phi * self.LeakyReLU(dloss_cphi - self.gamma)
        
        loss = loss_phi + loss_cphi + convergenceloss + divergenceloss + drloss 
        
        lossdict = self.generatorLoss.lossdict(
            loss=loss,
            drloss=drloss,
            loss_phi=loss_phi,
            dloss_phi=dloss_phi,
            loss_cphi=loss_cphi,
            dloss_cphi=dloss_cphi,
            divergenceloss=divergenceloss,
            convergenceloss=convergenceloss,
            Class=torch.tensor(float(cidx))
        )
        print(f'cidx={cidx}', lossdict)
        print('----nidx------->', nidx)
        self.vqgan.save_phi(concept, pathdir=self.pathdir, fname=f'final/concept.png')
        self.vqgan.save_phi(phi_sprime, pathdir=self.pathdir, fname=f'final/phi_sprime.png')
        self.vqgan.save_phi(phi_szegond, pathdir=self.pathdir, fname=f'final/phi_szegond.png')
        self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'final/phi.png')
        self.vqgan.save_phi(cphi, pathdir=self.pathdir, fname=f'final/cphi.png')
        assert False
        return loss, lossdict



class FUM_DR(FUM):
    def start(self, dr_vs_synthesis_flag=True):
        return super().start(dr_vs_synthesis_flag=False)
    
    # def on_train_epoch_end(self):
    #     cmatrix(self.v_ygrnt, self.v_ypred, f'/content/e0_val_cmat_before.png', normalize=False)
    #     cmatrix(self.t_ygrnt, self.t_ypred, f'/content/e0_train_cmat_before.png', normalize=False)
    #     assert False, 'END-TRAINING'

    def generator_step(self, batch, **kwargs):
        return super().generator_step__drcalgo(batch, **kwargs)
    
class FUM_Syn(FUM):
    def validation_step(self, batch, batch_idx, split='val'):
        return

    def training_step(self, batch, batch_idx, split='train'):
        return super().training_step__synalgo(batch, batch_idx, split='train')

    def generator_step(self, batch, **kwargs):
        return super().generator_step__synalgo(batch, **kwargs)
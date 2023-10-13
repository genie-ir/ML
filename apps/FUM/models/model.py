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

import albumentations as A 
from albumentations.pytorch import ToTensorV2
Transformer = A.Compose([
    A.Resize(256, 256),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
    ToTensorV2()
])

from utils.plots.plot1d import Plot1D
import os
import numpy as np
from PIL import Image

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
    # def call_init_from_ckpt(self):
    #     pass

    # NOTE: asli
    # def generator_step__drcalgo(self, batch, **kwargs):
    #     x = batch['X']
    #     batchsize = x.shape[0]
    #     x = (x / 127.5) - 1
        


    
    # NOTE: DR_CLASSIFIRE Training function.
    def normal_for_drc(self, xr1):
        xr1 = ((xr1 - (self.dr_classifire_normalize_mean * 255)) / (self.dr_classifire_normalize_std * 255)).detach()
        return xr1
    
    def check_dr(self, batch, split, batch_index, dr_pred):
        # batch['y_edit'] = (0 * torch.ones((batch['xs'].shape[0],), device=self.device)).long()
        
        
        # TODO uncomment it
        if split == 'train':
            self.t_ypred = self.t_ypred + list(dr_pred.argmax(dim=1).cpu().numpy())
            self.t_ygrnt = self.t_ygrnt + list(batch['y_edit'].cpu().numpy())
        else:
            self.v_ypred = self.v_ypred + list(dr_pred.argmax(dim=1).cpu().numpy())
            self.v_ygrnt = self.v_ygrnt + list(batch['y_edit'].cpu().numpy())
        
        
        loss = self.generator.ce(dr_pred, batch['y_edit'])
        
        if batch_index % 400 == 0:
            print('loss --->', loss.cpu().detach().item())
        return loss, dict(loss=loss.cpu().detach().item())
    
    def compute_loss(self, batch, clable, batchsize): # x is in class 0 
        # Q, qloss1 = self.vqgan.encode(x)
        # xrec1 = self.vqgan.decode(Q + self.generator.mac_class[clable-1](Q))
        zs = self.vqgan.phi2lat(batch['xs']).float().flatten(1)
        zc = self.vqgan.phi2lat(batch['xc'][clable-1]).float().flatten(1)

        Pbenuli = torch.sigmoid(self.generator.P).round()

        z = Pbenuli * zs + (1 - Pbenuli) * zc
        # signal_save(torch.cat([
        #     self.vqgan_fn_phi_denormalize(self.vqgan.lat2phi(zs)).detach(),
        #     self.vqgan_fn_phi_denormalize(self.vqgan.lat2phi(zc1)).detach(),
        #     self.vqgan_fn_phi_denormalize(self.vqgan.lat2phi(zc2)).detach()
        # ], dim=0), f'/content/test.png', stype='img', sparams={'chw2hwc': True, 'nrow': batchsize})

        z = self.generator.EncoderModel(z) # TODO
        # print('------------------>', self.generator.EncoderModel[0].linear1.weight[0,0])


        xrec1 = self.vqgan.lat2phi(z)
        xr1 = self.vqgan_fn_phi_denormalize(xrec1).detach()
        xr1 = ((xr1 - (self.dr_classifire_normalize_mean * 255)) / (self.dr_classifire_normalize_std * 255)).detach()
        xr1 = dzq_dz_eq1(xr1, xrec1)

        drloss1 = self.generator.cew(self.generator.softmax(self.dr_classifire(xr1)[0]), (clable * torch.ones((batchsize,), device=self.device)).long())
        aeloss1, log_dict_ae = self.vqgan.loss(0, batch['xs'], xrec1, 0, self.global_step, last_layer=self.vqgan.get_last_layer(), split="train")
        return xrec1, drloss1, aeloss1

    def generator_step__drcalgo(self, batch, **kwargs):
        drpred = self.generator.dr_classifire(
            self.normal_for_drc((batch['xs']+1) * 127.5)
        )[0].unsqueeze(0)
        print('dr_pred', drpred, drpred.shape)
        assert False
        return self.check_dr(
            batch, kwargs['split'], 
            kwargs['batch_idx'], 
            self.generator.softmax(
                drpred
            )
        )
        
        xrec1, drloss1, aeloss1 = self.compute_loss(batch, 1, batch['xs'].shape[0])
        xrec2, drloss2, aeloss2 = self.compute_loss(batch, 2, batch['xs'].shape[0])

        if kwargs['batch_idx'] % 400 == 0:
            Pbenuli = torch.sigmoid(self.generator.P).round()
            neon = Plot1D(xlabel='cluster', ylabel='selection for xs')
            neonc = Plot1D(xlabel='cluster', ylabel='selection for xc')
            neon.plot(range(256), Pbenuli.detach().cpu().numpy(), label=f'Adaptive filter', plot_type='step')
            neonc.plot(range(256), (1-Pbenuli).detach().cpu().numpy(), label=f'Adaptive filter', plot_type='step')
            neon.savefig(f'/content/filter_selection_style.png')
            neonc.savefig(f'/content/filter_selection_class.png')

            signal_save(torch.cat([
                (batch['xs']+1) * 127.5,
                self.vqgan_fn_phi_denormalize(xrec1).detach(),
                self.vqgan_fn_phi_denormalize(xrec2).detach()
            ], dim=0), f'/content/syn.png', stype='img', sparams={'chw2hwc': True, 'nrow': batch['xs'].shape[0]})
        
        loss = drloss1 + drloss2 + aeloss1 + aeloss2

        # assert False
        return loss, dict(loss=loss.cpu().detach().item())












































    def __2generator_step__drcalgo(self, batch, **kwargs):
        # phi = self.vqgan.lat2phi(batch['X'].flatten(1).float())
        # phi_denormalized = self.vqgan_fn_phi_denormalize(phi).detach()
        # signal_save(phi_denormalized, f'/content/a.png', stype='img', sparams={'chw2hwc': True})
        # phi_denormalized = self.get_eyepacs_data_batch(batch)
        
        
        
        x = batch['X']
        batchsize = x.shape[0]
        x = (x / 127.5) -1
        
        
        
        # if kwargs['batch_idx'] % 400 == 0:
        #     V, xrec1 = self.generator.vqgan.training_step_for_drc(x, self.generator.mac_class1)
        #     V, xrec2 = self.generator.vqgan.training_step_for_drc(x, self.generator.mac_class2)
        #     signal_save(torch.cat([
        #         (x+1) * 127.5,
        #         self.vqgan_fn_phi_denormalize(xrec1).detach(),
        #         self.vqgan_fn_phi_denormalize(xrec2).detach()
        #     ], dim=0), f'/content/syn.png', stype='img', sparams={'chw2hwc': True, 'nrow': batchsize})
        #     return


        if kwargs['batch_idx'] % 2 == 0:
            total_loss = self.generator.vqgan.training_step_for_drc(
                x, self.generator.mac_class1,
                torch.ones((batchsize,), device=self.device).long(),
                self.generator.softmax, self.generator.ce, self.dr_classifire,
                kwargs['batch_idx'], 1
            )
        else:
            total_loss = self.generator.vqgan.training_step_for_drc(
                x, self.generator.mac_class2,
                (2 * torch.ones((batchsize,), device=self.device)).long(),
                self.generator.softmax, self.generator.ce, self.dr_classifire,
                kwargs['batch_idx'], 2
            )
            
        return total_loss, dict(loss=total_loss.cpu().detach().item())
            # x1 = self.vqgan_fn_phi_denormalize(xrec1).detach()
            # x1 = dzq_dz_eq1(x1, xrec1)
            # x1 = (x1 - (self .dr_classifire_normalize_mean * 255)) / (self.dr_classifire_normalize_std * 255)
            
            
            
            # output1 = self.dr_classifire(x1)[0]#.detach()
            # dr_pred1 = self.generator.softmax(output1)
            # drloss1 = self.generator.ce(dr_pred1, torch.ones((batchsize,), device=self.device).long())
            # vaeloss = vaeloss1
            # drloss = drloss1
        # else:
        #     vaeloss2, xrec2 = self.generator.vqgan.training_step_for_drc(x, self.generator.mac_class2)
        #     x2 = self.vqgan_fn_phi_denormalize(xrec2).detach()
        #     x2 = dzq_dz_eq1(x2, xrec2)
        #     x2 = (x2 - (self.dr_classifire_normalize_mean * 255)) / (self.dr_classifire_normalize_std * 255)
        #     output2 = self.dr_classifire(x2)[0]#.detach()
        #     dr_pred2 = self.generator.softmax(output2)
        #     drloss2 = self.generator.ce(dr_pred2, (2 * torch.ones((batchsize,), device=self.device)).long())
        #     drloss = drloss2
        #     vaeloss = vaeloss2
        
        # loss = vaeloss + drloss

        # if kwargs['split'] == 'train':
        #     self.t_ypred = self.t_ypred + list(dr_pred.argmax(dim=1).cpu().numpy())
        #     self.t_ygrnt = self.t_ygrnt + list(batch['y_edit'].cpu().numpy())
        # else:
        #     self.v_ypred = self.v_ypred + list(dr_pred.argmax(dim=1).cpu().numpy())
        #     self.v_ygrnt = self.v_ygrnt + list(batch['y_edit'].cpu().numpy())
        # return loss, dict(loss=loss.cpu().detach().item())
        # return drloss, dict(drloss=drloss.cpu().detach().item())

    def getbatch(self, batch):
        # return super().getbatch(batch, skey='Xidx')
        # return super().getbatch(batch, skey='X')
        return super().getbatch(batch, skey='x')
    
    
    
    
    
    
    
    
    # NOTE: Synthesis Algorithm.
    def __training_step__synalgo(self, batch, batch_idx, split='train'):
        # if batch_idx == 0:
        #     print('-'*60)
        #     print(self.generator.scodebook.embedding.weight)
        #     print('-'*60)
        # B = torch.tensor([2,3], device=self.device)
        # print(f'iter{batch_idx} | before', self.generator.scodebook.embedding.weight[B[0],0], self.generator.scodebook.embedding.weight[B[0],0].exp())

        B = batch['Xidx']
        for cidx in range(self.nclasses):
            # print('----grad---->', self.generator.scodebook.embedding.weight.grad)
            batch['cidx'] = cidx
            batch['bidx'] = batch_idx
            batch[self.signal_key] = self.generator.scodebook.fwd_nbpi(B).exp() #.clone()
            super().training_step(batch, batch_idx, split)
        # print(f'iter{batch_idx} | after', self.generator.scodebook.embedding.weight[B[0],0], self.generator.scodebook.embedding.weight[B[0],0].exp())
        # if batch_idx == 2:
        #     assert False, batch_idx
    
    
    def start(self, dr_vs_synthesis_flag=True):
        self.gamma = - 0.1
        self.vqgan_dataset = '/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all/data/eyepacs_all_ims'

        self.generator.ce = nn.CrossEntropyLoss()
        self.dr_classifire_normalize_std = torch.tensor([0.1252, 0.0857, 0.0814]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to('cuda')
        self.dr_classifire_normalize_mean = torch.tensor([0.3771, 0.2320, 0.1395]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to('cuda')

        self.vqgan_fn_phi_denormalize = lambda G: ((((G.clamp(-1., 1.))+1)/2)*255)#.transpose(0,1).transpose(1,2)
        self.generator.softmax = torch.nn.Softmax(dim=1)
        self.t_ypred = []
        self.t_ygrnt = []
        self.v_ypred = []
        self.v_ygrnt = []
        self.qshape = (self.qch, self.qwh, self.qwh)
        # self.vseg = makevaslsegmentation('/content/drive/MyDrive/storage/dr_classifire/unet-segmentation/weight_retina.hdf5')

        # self.dr_classifire, cfg = makeDRclassifire('/content/drive/MyDrive/storage/dr_classifire/best_model.pth')
        # self.dr_classifire = self.dr_classifire.to('cuda')
        # self.dr_classifire.requires_grad_(False) # delete
        
        # self.generator.mac_class1 = MAC(fwd='fConv2d', ch=256)
        # self.generator.mac_class2 = MAC(fwd='fConv2d', ch=256)
        # self.generator.mac_class = [
        #     self.generator.mac_class1,
        #     self.generator.mac_class2
        # ]



        # self.generator.mac_class1 = MAC(units=2, shape=self.qshape)
        # self.generator.mac_class2 = MAC(units=2, shape=self.qshape)
        # self.generator.vqgan = self.vqgan
        # del self.vqgan
        # self.generator.vqgan.requires_grad_(True)

        # if dr_vs_synthesis_flag:
        #     self.hp('lambda_loss_scphi', (list, tuple), len=self.nclasses)
        #     self.hp('lambda_drloss_scphi', (list, tuple), len=self.nclasses)
            
        #     self.phi_shape = (self.phi_ch, self.phi_wh, self.phi_wh)
        #     self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)
        #     # self.generator.scodebook = VectorQuantizer(ncluster=self.ncluster, dim=self.latent_dim, zwh=1)
        #     # self.generator.mac = nn.Sequential(*[
        #     #     MAC(units=2, shape=self.qshape) for c in range(self.nclasses)
        #     # ])
        #     # self.generator.mac_class1 = MAC(units=2, shape=self.qshape)
        #     # self.generator.mac_class2 = MAC(units=2, shape=self.qshape)
        #     self.dr_classifire.requires_grad_(False)
        # else:
        #     pass
        #     # self.generator.dr_classifire = self.dr_classifire

    def __c2phi(self, cross, tag='', phi_concept=None, phiName='fundus'):
        # list_of_distance_to_mode = []
        # BASIC the very basic code of idea behind chaining concept.
        # phi = self.vqgan.lat2phi(cross)
        # sn = self.vqgan.phi2lat(phi).flatten(1).float().detach()
        # return phi, sn

        # cross = cross[0].unsqueeze(0) #TODO: just for test you can use 1 sample per batch.

        # PHI = []
        # PHI_L = []
        
        # IDEA s1 = phi0 # is better than --> torch.zeros((batch_size,) + self.phi_shape, device=self.device) --> becuse the first one is diffrentiable. NOTE: each time you must do: s1=s1+phi
        Q = self.vqgan.lat2qua(cross)
        phi0 = self.vqgan.qua2phi(Q)
        nl = self.vqgan.phi2lat(phi0.detach()).float()
        # _np = phi0.detach()
        for N in range(1, self.phi_steps):
            # list_of_distance_to_mode.append(nl.flatten(1))
            np = self.vqgan.lat2phi(nl)
            # print(f'({N-1},{N})-------ssim-------->', SSIM(_np, np))
            nnl = self.vqgan.phi2lat(np).float()
            # qe_mse = ((nl-nnl)**2).mean()
            qe_mse = ((nl-nnl).abs()).sum()



            # if N %2 == 0:
            #     PHI.append(np)
            #     PHI_L.append(qe_mse.item())



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
        
        
        # self.vqgan.save_phi(torch.cat(PHI, dim=0), pathdir=self.pathdir, fname=f'PHI_{phiName}.png', nrow=4)
        # neon = Plot1D(xlabel='Iteration', ylabel='Quantization Error | sum_absolute_error(z*[i], z*[i-1])')
        # neon.plot(range(len(PHI_L)), PHI_L, label=f'convergence curve')
        # neon.savefig(f'/content/convergence_{phiName}.png')
        return (phi0, Q), sn, np
    
    def get_eyepacs_data_batch(self, batch):
        B = []
        for b in batch['x']:
            bb = os.path.split(b)[1].replace('.npy', '.jpeg')
            bbb = np.array(Image.open(os.path.join(self.vqgan_dataset, bb))).astype(np.uint8)
            # bbb = (bbb/127.5 - 1.0).astype(np.float32)
            # bbb = Transformer(image=bbb)['image']
            bbb = Transformer(image=bbb)['image'].unsqueeze(0)
            B.append(bbb)
            # print(bbb.shape, bbb.min(), bbb.max())
        F = torch.cat(B, dim=0).to(self.device)
        # signal_save(F, f'/content/F.png', stype='img', sparams={'chw2hwc': True})
        return F
    
    def __generator_step__synalgo(self, batch, **kwargs):
        bidx = batch['bidx'] 
        cidx = batch['cidx']
        batch['y_edit'] = torch.tensor([cidx, cidx], device=self.device) # NOTE: batch size is 2 :)
        ln = batch[self.signal_key]

        (phi, q_phi), sn, concept = self.__c2phi(ln, phiName='random') # NOTE `sn` and `concept` doesnt have derevetive.
        # (phi, q_phi), sn, concept = self.__c2phi(batch['X0'].flatten(1).float(), phiName='orginal') # DELETE: this line is just for test and it must be delete after.
        
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
        dloss_cphi = -torch.mean(self.vqgan.loss.discriminator(cphi)) # NOTE DLOSS.shape=(B,1,30,30) float32.
        
        loss_phi = dloss_phi
        loss_cphi = dloss_cphi
        # loss_phi = self.lambda_loss_phi * self.LeakyReLU(dloss_phi - self.gamma)
        # loss_cphi = self.lambda_loss_phi * self.LeakyReLU(dloss_cphi - self.gamma)
        
        # loss = loss_phi + loss_cphi + convergenceloss + divergenceloss + drloss 
        loss = loss_cphi + convergenceloss + divergenceloss + drloss 
        
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
        if bidx % 500 == 0:
            print(f'cidx={cidx} | bidx={bidx}', lossdict)
            self.vqgan.save_phi(concept, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/concept.png')
            self.vqgan.save_phi(phi_sprime, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/phi_sprime.png')
            self.vqgan.save_phi(phi_szegond, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/phi_szegond.png')
            self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/phi.png')
            self.vqgan.save_phi(cphi, pathdir=self.pathdir, fname=f'final/{bidx}-{cidx}/cphi.png')
        return loss, lossdict

    def generator_step__synalgo(self, batch, **kwargs):
        batch['X'] = (batch['X']/127.5 - 1.0)
        latent = self.vqgan.phi2lat(batch['X']).flatten(1).float()
        quant = self.vqgan.lat2qua(latent)
        xrec = self.vqgan.qua2phi(quant)
        print(batch['X'].shape, batch['X'].min(), batch['X'].max(), batch['X'].dtype)
        print('L', latent.shape)
        print('Q', quant.shape)
        print('xrec', xrec.shape)
        signal_save((batch['X'] +1)*127.5, f'/content/F.png', stype='img', sparams={'chw2hwc': True})
        self.vqgan.save_phi(xrec, pathdir=self.pathdir, fname=f'/content/R.png')
        assert False
        # bidx = batch['bidx'] 
        # cidx = batch['cidx']
        # batch['y_edit'] = torch.tensor([cidx, cidx], device=self.device) # NOTE: batch size is 2 :)
        # ln = batch[self.signal_key]

class FUM_DR(FUM):
    def call_init_from_ckpt(self):
        pass
    def start(self, dr_vs_synthesis_flag=True):
        super().start(dr_vs_synthesis_flag=False)
        from torchvision.models import vgg16
        self.generator.dr_classifire = vgg16(pretrained=True)
        self.generator.dr_classifire.classifier[-1] = nn.Linear(in_features=4096, out_features=3)
        self.generator.dr_classifire = self.generator.dr_classifire.to('cuda')
        # print(self.generator.dr_classifire )
        # self.generator.dr_classifire, cfg = makeDRclassifire('/content/drive/MyDrive/storage/dr_classifire/best_model.pth')
        # self.generator.dr_classifire = self.generator.dr_classifire.to('cuda')
        
        
        
        # print('before', self.generator.dr_classifire.classifier1[0].weight[10, :10])
        # self.generator.dr_classifire.requires_grad_(False) # delete
        # self.init_from_ckpt(
        #     # '/content/drive/MyDrive/storage/ML_Framework/FUM/logs/2023-10-11T21-37-15_svlgan_dr/checkpoints/e450.ckpt'
        #     # '/content/drive/MyDrive/storage/ML_Framework/FUM/logs/2023-10-12T14-34-38_svlgan_dr/checkpoints/e300pretrain.ckpt'
        # )
        # print('after', self.generator.dr_classifire.classifier1[0].weight[10, :10])
        # self.dr_weight = torch.tensor([1, 1.5 ,9.4], dtype=torch.float32).to('cuda')
        # self.generator.cew = nn.CrossEntropyLoss(weight=self.dr_weight)
        # assert False
        
        


        # model_dict = self.state_dict()
        # ckpt = '/content/drive/MyDrive/storage/ML_Framework/FUM/logs/2023-09-22T12-23-17_svlgan_dr/checkpoints/last.ckpt'
        # i=15
        # print('before', self.generator.dr_classifire.classifier3[0].weight[i,:5])
        # sd = torch.load(ckpt, map_location='cpu')['state_dict']
        # sd = {k: v for k, v in sd.items() if k in model_dict}
        # model_dict.update(sd)
        # self.load_state_dict(sd)
        # print('after', self.generator.dr_classifire.classifier3[0].weight[i,:5])
        # self.vqgan.init_from_ckpt('/content/drive/MyDrive/storage/ML_Framework/VQGAN_OK/logs/2023-10-01T21-31-26_eyepacs_vqgan/checkpoints/lastV6.ckpt')
        # print('after2', self.generator.dr_classifire.classifier3[0].weight[i,:5])
        # assert False
        # self.vqgan.init_from_ckpt('/content/drive/MyDrive/storage/ML_Framework/VQGAN_OK/logs/2023-10-01T21-31-26_eyepacs_vqgan/checkpoints/lastV6.ckpt')
        
        
        # TODO uncumment
        # self.generator.testvar = nn.Parameter(torch.randn((1,)))
        # self.generator.P = nn.Parameter(torch.randn((256,)))

        # self.generator.EncoderModel = torch.nn.Sequential(
        #     *[nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True) for n in range(8)]
        # )
        
        
    # def validation_step(self, batch, batch_idx, split='val'):
    #     self.generator.dr_classifire.train()
    #     torch.set_grad_enabled(True)
    #     super().training_step(batch, batch_idx, 'train')
    #     self.generator.dr_classifire.eval()
    #     torch.set_grad_enabled(False)
    #     return super().validation_step(batch, batch_idx, split)

    def on_train_epoch_end(self):
        self.v_ygrnt = self.v_ygrnt + [1,2]
        self.t_ygrnt = self.t_ygrnt + [1,2]
        self.v_ypred = self.v_ypred + [1,2]
        self.t_ypred = self.t_ypred + [1,2]
        cmatrix(self.v_ygrnt, self.v_ypred, f'/content/e0_val_cmat_before.png', normalize=False)
        cmatrix(self.t_ygrnt, self.t_ypred, f'/content/e0_train_cmat_before.png', normalize=False)
        assert False, 'END-TRAINING'

    def validation_step(self, batch, batch_idx, split='val'):
        return
    def generator_step(self, batch, **kwargs):
        return super().generator_step__drcalgo(batch, **kwargs)
    
class FUM_Syn(FUM):
    def validation_step(self, batch, batch_idx, split='val'):
        return

    # def training_step(self, batch, batch_idx, split='train'):
    #     return super().training_step__synalgo(batch, batch_idx, split='train')

    def generator_step(self, batch, **kwargs):
        return super().generator_step__synalgo(batch, **kwargs)
import torch
import torch.nn as nn
from loguru import logger
import torchvision
import torch.nn.functional as F
from apps.VQGAN.modules.losses.lpips import LPIPS
from apps.VQGAN.modules.discriminator.model import NLayerDiscriminator
from libs.basicIO import signal_save
from utils.pt.tricks.gradfns import dzq_dz_eq1


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if isinstance(threshold, int) and (global_step < threshold):
        weight = value
    elif isinstance(threshold, bool) and (not threshold):
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        # from dependency.BCDU_Net.Retina_Blood_Vessel_Segmentation.pretrain import pretrain as makevaslsegmentation
        # self.vseg = makevaslsegmentation('/content/drive/MyDrive/storage/dr_classifire/unet-segmentation/weight_retina.hdf5')
        # self.vqgan_fn_phi_denormalize = lambda G: ((((G.clamp(-1., 1.))+1)/2)*255)#.transpose(0,1).transpose(1,2)


        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval() # VGG16
        self.perceptual_weight = perceptual_weight



        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm, ndf=disc_ndf)
        self.discriminator_large = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm, ndf=disc_ndf, kw=9)
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param_fidx in [26, 28]:
            for param in self.vgg16.features[param_fidx].parameters():
                param.requires_grad = True
        n_inputs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256)
        )
        self.vgg16_head = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.Sigmoid()
        )





        
        
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        logger.info(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        self.start()

    def start(self):
        # print(self.discriminator)
        # assert False
        self.eps = torch.tensor(-5).exp() # tensor(0.0067)
        pass
    
    def vgg16head_mean(self, x):
        disc = self.vgg16_head(x).mean()

        DlossCorrect = disc.detach().clamp(self.eps)
        DlossCorrect = dzq_dz_eq1(DlossCorrect, disc)
        return DlossCorrect

    def omega_of_phi(self, x):
        return self.vgg16head_mean(self.vgg16(x))
    
    def omega_of_phi_givvenRo(self, ro):
        return self.vgg16head_mean(ro)
    
    def Ro(self, x):
        return self.vgg16(x)
    
    
    def D1(self, x): # discriminator
        disc = (self.discriminator(x.contiguous())).mean()
        DlossCorrect = disc.detach().clamp(self.eps)
        DlossCorrect = dzq_dz_eq1(DlossCorrect, disc)
        return DlossCorrect
    def D2(self, x): # discriminator_large
        disc = (self.discriminator_large(x.contiguous())).mean()
        DlossCorrect = disc.detach().clamp(self.eps)
        DlossCorrect = dzq_dz_eq1(DlossCorrect, disc)
        return DlossCorrect
    
    def D12(self, x, l1=1, l2=1, flag=False, split=''):
        d1 = self.D1(x) # 0 -> exp(-5) <= d1 <=1
        d2 = self.D2(x) # 0 -> exp(-5) <= d2 <=1
        if flag:
            d1 = 1 - d1
            d2 = 1 - d2
        d1 = l1 * (-1 * (d1.log()))
        d2 = l2 * (-1 * (d2.log()))
        loss = d1 + d2

        log = {
            "{}/loss".format(split): loss.clone().detach().mean().item(),
            "{}/d1".format(split): d1.clone().detach().mean().item(),
            "{}/d2".format(split): d2.clone().detach().mean().item(),
        }
        return loss, log

    def geometry(self, grandtrouth, prediction, split, pw=0, recln1p=False): # pw=0.1
        rec_loss = torch.abs(grandtrouth.contiguous() - prediction.contiguous())
        if recln1p:
            rec_loss = (1+rec_loss).log()
        
        if pw > 0:
            p_loss = pw * self.perceptual_loss(grandtrouth.contiguous(), prediction.contiguous())
        else:
            p_loss = torch.tensor(0.0)

        loss = (rec_loss + p_loss).mean()
        log = {
            "{}/loss".format(split): loss.clone().detach().mean().item(),
            "{}/rec_loss".format(split): rec_loss.clone().detach().mean().item(),
            "{}/p_loss".format(split): p_loss.clone().detach().mean().item(),
        }
        return loss, log
    

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        # retain_graph = True
        retain_graph = False
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=retain_graph)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=retain_graph)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=retain_graph)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=retain_graph)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, cond=None, dw=0.1, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss # what is p_loss shape? is scaller?
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss) # TODO total reconstruction loss contains norm1 and vgg perseptual loss.
        # print('nll_loss', nll_loss, nll_loss.shape, nll_loss.dtype) # nll_loss tensor(0.1391, grad_fn=<MeanBackward0>) torch.Size([]) torch.float32
        # NOTE: reconstructionloss(norm1 and vgg_perseptual) -> done!

        # now the GAN part
        if optimizer_idx == 0: # reconstruction/generator
            # generator update
            if dw > 0:
                if cond is None:
                    # print('HEREEEEEEEEEEEEEEEEEEEEE')
                    logits_fake = torch.log(self.discriminator(reconstructions.contiguous()))
                    logits_fake_large = torch.log(self.discriminator_large(reconstructions.contiguous()))
                else:
                    assert self.disc_conditional
                    assert False
                    # logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
                
                # print('!!!!!!!logits_fake', logits_fake.shape, logits_fake.dtype, logits_fake.min().item(), logits_fake.max().item(), logits_fake.mean().item()) # !!!!!!!logits_fake torch.Size([1, 1, 30, 30]) torch.float32 -0.9964276552200317 -0.4953823685646057 -0.7139382362365723
                # print('!!!!!!!logits_fake_large', logits_fake_large.shape, logits_fake_large.dtype, logits_fake_large.min().item(), logits_fake_large.max().item(), logits_fake_large.mean().item()) # !!!!!!!logits_fake_large torch.Size([1, 1, 15, 15]) torch.float32 -9.900166511535645 -1.3470740668708459e-05 -1.919838786125183
                g_loss = -torch.mean(logits_fake)
                g_loss_large = -torch.mean(logits_fake_large)
                gloss_total = dw * (g_loss + g_loss_large)
                # print('$$$$$$', g_loss, g_loss_large) # $$$$$$ tensor(0.7139, grad_fn=<NegBackward0>) tensor(1.9198, grad_fn=<NegBackward0>)
                print('@@@@@@@@ gloss_total', gloss_total, gloss_total.dtype, gloss_total.shape)
                # NOTE: multiscale disc loss -> done!
            else:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                gloss_total = 0

            # try:
            #     d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            # except RuntimeError as e_RuntimeError:
            #     # logger.critical(e_RuntimeError)
            #     assert not self.training
            #     # d_weight = torch.tensor(0.0)
            #     d_weight = torch.tensor(1.0)

            # disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start) # here value is 0 :|
            # loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
            loss = nll_loss + gloss_total

            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
            #    "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
            #    "{}/rec_loss".format(split): rec_loss.detach().mean(),
            #    "{}/p_loss".format(split): p_loss.detach().mean(),
            #    "{}/d_weight".format(split): d_weight.detach(),
            #    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/gloss_total".format(split): gloss_total.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1: # disc loss
            # second pass for discriminator update
            if cond is None:
                print('DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD')
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                assert False
                # logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                # logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            print('@@ real', logits_real.shape, logits_real.dtype, logits_real.min().item(), logits_real.max().item(), logits_real.mean().item())
            print('@@ fake', logits_fake.shape, logits_fake.dtype, logits_fake.min().item(), logits_fake.max().item(), logits_fake.mean().item())
            assert False

            # disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            # d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            d_loss = self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }
            return d_loss, log

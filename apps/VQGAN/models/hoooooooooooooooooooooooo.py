import torch
import os
import numpy as np
import torchvision
from os.path import join, exists
import pathlib
from einops import rearrange
from PIL import Image

def rootPath():
    return pathlib.Path(__file__).parents[1]

def pathBIO(fpath: str, **kwargs):
    if fpath.startswith('//'):
        fpath = join(rootPath(), fpath[2:])
    return fpath

def __signal_save__img_Tensor(images, fpath, nrow=None, fn=None, sreturn=False, chw2hwc=False):
        if fn is None:
            fn = lambda G: G

        nrow = images.shape[0] if nrow is None else nrow

        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()

        grid = torchvision.utils.make_grid(images, nrow=nrow) # this grid finally contains table of iamges like this -> [images[k].shape[0]/nrow, nrow] ; Notic: grid is tensor with shape: ch x h? x w?

        if chw2hwc:
            grid = rearrange(grid, 'c h w -> h w c').contiguous()

        grid = fn(grid).numpy().astype(np.uint8)
        signal_save(grid, fpath)

        if sreturn:
            imgs = fn(images)
            imgs = rearrange(imgs, 'c h b w -> b c h w').contiguous()
            return imgs

def signal_save(s, path, makedirsFlag=True, stype=None, sparams=None):
    sparams = dict() if sparams is None else sparams
    assert isinstance(sparams, dict), '`type(sparams)={}` | It must be dict | looking this maybe useful: `sparams={}`'.format(type(sparams), sparams)
    path = pathBIO(path)
    dpath, fname = os.path.split(path)
    fname_lower = fname.lower()
    if makedirsFlag:
        os.makedirs(dpath, exist_ok=True)

    if isinstance(s, torch.Tensor):
        if stype == 'img':
            return __signal_save__img_Tensor(s, path, **sparams)
        s = s.cpu().detach().numpy()

    if isinstance(s, np.ndarray): # image signal
        if any(ext.lower() in fname_lower for ext in ['.png', '.jpg', '.jpeg']):
            return Image.fromarray(s).save(path)
        if any(ext.lower() in fname_lower for ext in ['.npy']):
            return np.save(path, s)

def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size # bytes


def bb(img, color='green'):
    # print(img.shape)
    return torchvision.utils.draw_bounding_boxes(((img.squeeze()+1)*127.5).to(torch.uint8), torch.tensor([0,0, 255,255], dtype=torch.int).unsqueeze(0), colors=color).unsqueeze(0) /127.5 -1


import albumentations as A
from albumentations.pytorch import ToTensorV2

dr_transformer0 = A.Compose([
    ToTensorV2() # hxwxch -> chxhxw
])
dr_transformer1 = A.Compose([
    ToTensorV2() # hxwxch -> chxhxw
])
def f(p, flag=True):
  out = dr_transformer0(image=np.array(Image.open(p)))['image'].unsqueeze(0)
  if flag:
    out = out / 127.5 - 1
  return out

Repo = {}

def g(p, q):
  p0=p.split('/')[-1].replace('_clahe', '').replace('.jpg', '')
  p1=f'RetinaLessions/{p0}/{q}/'
  R = {}
  for r, fi in [['cvh', 1], ['fmask', 1], ['fundus', 0], ['lesion', 0], ['lmask', 1]]:
    func = eval(f'dr_transformer{fi}')
    R[r] = f(f'{p1}/{r}.jpg', fi==0)
  R['asli'] = f(p)
  R['asli_lmask'] = f(p.replace('fundus', 'lmask'), False) / 255
  R['asli_fmask'] = f(p.replace('fundus', 'fundus-mask'), False) / 255
  R['union'] = ((R['lmask']/255) + (R['asli_lmask'])) - ((R['lmask']/255) * (R['asli_lmask']))
  R['sorakh'] = (((R['asli']+1)*127.5) * (1 - R['union'] )  ) / 127.5 - 1
  R['sonly'] = (((R['asli']+1)*127.5) * (1 - R['asli_lmask'] )  ) / 127.5 - 1
  R['final'] = (((R['sorakh']+1)*127.5) + (R['union'] * R['asli_fmask'] * R['lmask']/255 * ((R['fundus']+1)*127.5))) / 127.5 - 1
  Repo[f'{p0}_{q}'] = R
  return Repo[f'{p0}_{q}']['fundus']


def h(p, q):
  p0=p.split('/')[-1].replace('_clahe', '').replace('.jpg', '')
  p1=f'RetinaLessions/{p0}/{q}/'
  return Repo[f'{p0}_{q}']['sonly']


























P0 = 'prototype/fundus/1/10192_left.jpg'
P1 = 'prototype/fundus/2/14651_right_clahe.jpg'
P2 = 'prototype/fundus/4/2800_left_clahe.jpg'

imglogger=[
    dict(
        xc0=g(P0, '2'),
        xc1=g(P0, '[34]'),
        xs=f(P0),
        c0_optidx0_pipline=dict(
            psis_tp_final=h(P0, '2')
        ),
        c1_optidx0_pipline=dict(
            psis_tp_final=h(P0, '[34]')
        )
    ),

    dict(
        xc0=g(P1, '[01]'),
        xc1=g(P1, '[34]'),
        xs=f(P1),
        c0_optidx0_pipline=dict(
            psis_tp_final=h(P1, '[01]')
        ),
        c1_optidx0_pipline=dict(
            psis_tp_final=h(P1, '[34]')
        )
    ),

    dict(
        xc0=g(P2, '[01]'),
        xc1=g(P2, '2'),
        xs=f(P2),
        c0_optidx0_pipline=dict(
            psis_tp_final=h(P2, '[01]')
        ),
        c1_optidx0_pipline=dict(
            psis_tp_final=h(P2, '2')
        )
    ),
]





for krepo in Repo:
  img = Repo[krepo]['sonly']
  signal_save((torch.cat([
      img
  ], dim=0)+1)*127.5, f'/content/stack/s/{krepo}.png', stype='img', sparams={'chw2hwc': True, 'nrow': 5})




signal_save((torch.cat([
    imglogger[0]['xc0'], imglogger[0]['xc1'], bb(imglogger[0]['xs']), bb(imglogger[0]['c0_optidx0_pipline']['psis_tp_final'], 'red'), bb(imglogger[0]['c1_optidx0_pipline']['psis_tp_final'], 'red'),
    imglogger[1]['xc0'], imglogger[1]['xc1'], bb(imglogger[1]['c0_optidx0_pipline']['psis_tp_final'], 'blue'), bb(imglogger[1]['xs']), bb(imglogger[1]['c1_optidx0_pipline']['psis_tp_final'], 'red'),
    imglogger[2]['xc0'], imglogger[2]['xc1'], bb(imglogger[2]['c0_optidx0_pipline']['psis_tp_final'], 'blue'), bb(imglogger[2]['c1_optidx0_pipline']['psis_tp_final'], 'blue'), bb(imglogger[2]['xs'])
], dim=0)+1)*127.5, f'/content/res.png', stype='img', sparams={'chw2hwc': True, 'nrow': 5})
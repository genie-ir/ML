import torch
import torch.nn as nn
from libs.basicDS import retrieve
from libs.coding import md5 as md5fn
from utils.tools import get_ckpt_path as get_ckpt_path_fn

URL_MAP = {
    'vgg_lpips': 'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1'
}

CKPT_MAP = {
    'vgg_lpips': 'vgg.pth'
}

MD5_MAP = {
    'vgg_lpips': 'd507d7349b931f0638a25a48a722f98a'
}

def get_ckpt_path(name, root, check=False):
    return get_ckpt_path_fn(name, root, check=check,
        URL_MAP=URL_MAP, MD5_MAP=MD5_MAP, CKPT_MAP=CKPT_MAP)

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


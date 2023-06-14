import torch
from utils.pt.distance import L2S

# TODO: check [d(veqQuantizer) / d(X)] and [d(veqQuantizer) / d(C)]

def veqQuantizer(X, C, output_shape=None, f=None):
    """
        veqQuantizer assumtion: X.shape=np x d, C.shape=nc x d 
        f is distance function By default Square of Norm2 is considered.

    Example:
        d = 2 # 64 latent space
        np, nc = 8, 4 # num_points=16000, num_clusters=300
        X = torch.randint(1,9, (np, d), dtype=torch.float32)
        C = torch.randint(1,9, (nc, d), dtype=torch.float32)
        quantized = veqQuantizer(X, C)
        print(quantized, quantized.shape)
    """
    device, nc = X.device, C.shape[0]
    f = L2S if f is None else f

    distances = f(X, C)
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    encodings = torch.zeros(encoding_indices.shape[0], nc, device=device)
    encodings.scatter_(1, encoding_indices, 1)
    quantized = torch.matmul(encodings, C)
    
    return quantized

def veqQuantizerImg(X, C, **kwargs):
    """
        X.shape=(B, CH, H, W) -> (B, H, W, CH) -> (B*H*W, CH)
        C.shape=(nc, d)
        notice: CH == d (is True)
    Example:
        d, nc = 64, 300
        X = torch.randint(1,9, (8, d, 32, 32), dtype=torch.float32)
        C = torch.randint(1,9, (nc, d), dtype=torch.float32)
        X.requires_grad = True
        C.requires_grad = True
        Q, Qp, Xp = veqQuantizerImg(X, C)
        L = utils.loss.vqvae_loss(Xp, Q)
        print(L)
    """
    CH = X.shape[1]
    Xp = X.permute(0, 2, 3, 1).contiguous() # (B,CH,H,W) -> (B,H,W,CH)
    Xpshape = Xp.shape
    Xpflat = Xp.view(-1, CH) # (B,H,W,CH) -> (B*H*W, CH)
    Q = veqQuantizer(Xpflat, C, **kwargs).view(Xpshape) # (B*H*W, CH) -> (B,H,W,CH)
    Qp = Q.permute(0, 3, 1, 2).contiguous() # (B,H,W,CH) -> (B,CH,H,W)
    return Q, Qp, Xp # Qp is our interest. Notic: we will compute loss between Xp and Q.
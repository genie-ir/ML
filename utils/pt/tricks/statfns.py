import torch

def correlation(x, y=None, kill_dig=False):
    if y is None:
        if kill_dig == False:
            return torch.corrcoef(x)
        else:
            return torch.corrcoef(x) * (1-torch.eye(x.shape[0])).detach()
    
    # torch.zeros()
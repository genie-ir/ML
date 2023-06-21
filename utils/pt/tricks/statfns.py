import torch
import numpy as np

def correlation(x, y=None, kill_itself=False):
    if y is None:
        if kill_itself == False:
            return torch.corrcoef(x)
        else:
            return torch.corrcoef(x) * (1-torch.eye(x.shape[0])).detach()
    
    # Not diffrantiable (currently)
    out = []
    for i in range(x.shape[0]):
        out.append(torch.corrcoef(torch.cat([x[i:i+1, :], y], dim=0))[0, :].detach().numpy())
    return torch.tensor(np.array(out))
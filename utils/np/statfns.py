import numpy as np

def correlation(x, y=None, kill_itself=False):
    if y is None:
        if kill_itself == False:
            return np.corrcoef(x)
        else:
            return np.corrcoef(x) * (1-np.eye(x.shape[0]))
    
    out = []
    for i in range(x.shape[0]):
        out.append(np.corrcoef(np.concatenate([x[i:i+1, :], y], 0))[0, :][1:])
    return np.array(out)
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

def smoothing(x, y, smooth_dpi=300, smooth_k=3):
    xnew = np.linspace(x.min(), x.max(), smooth_dpi) # smooth_dpi represents number of points to make between T.min and T.max
    spl = make_interp_spline(x, y, k=smooth_k)  # type: BSpline
    y = spl(xnew)
    x = xnew
    return x, y
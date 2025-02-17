import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

def dzq_dz_eq1(zq, z, w=1):
    """
        transfer gradients from `zq` to `z`  | (zq -> z)
        `zq` and `z` must be the same shape
        (Notic): zq not change in terms of numerically but here we define a drevative path from zq to z such that (dzq/dz = 1)
        Example: 
            zq = dzq_dz_eq1(zq, z)
    """
    return (w * z) + (zq - (w * z)).detach()

######################################################################################

def round_with_grad(x):
    return x + ((x - x.long()).round() - (x - x.long())).detach()

######################################################################################

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None

def clamp_with_grad(input, min=None, max=None):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

######################################################################################

def onehot_with_grad(x, num_classes: int):
    """
        x is free to have any shape
        Example: 
            t1 = onehot_with_grad(x, N)
            t1.retain_grad()
            y = t1 @ e # t1 * 6
            y.retain_grad()
            print(y, y.grad, t1, t1.grad, x, x.grad)
        
        Example(usecase for embedding layer):
            y = onehot_with_grad(x, number_of_classes) @ e.weight          # insetead of this  -> [  y = e(x) #here only dy/de.weight is available ]
            but here dy/dx is available and dy/de.weight is available
    """
    t = round_with_grad(x).unsqueeze(-1)
    t1 = clamp_with_grad(t, 1, num_classes-1) * F.one_hot(t.squeeze().round().long().abs().clamp(0, num_classes-1), num_classes=num_classes).detach() 
    return clamp_with_grad(t1, 0, 1)

######################################################################################


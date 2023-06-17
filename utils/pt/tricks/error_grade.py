import torch

def error_grade(ground_truth, n=0):
    """
        pred = ground_truth + erorr
        what is the `grade` of the `error`? 
        the `grade` of the `error` is `min` `n` such that `pred` tensor is be a good prediction of ground truth.
    """
    xfN = ((torch.randint(0, 2, ground_truth.shape, device=ground_truth.device) - 0.5) * 2) / (10 ** n)
    return ground_truth + xfN
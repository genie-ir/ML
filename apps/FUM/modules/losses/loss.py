try:
    from utils.pt.lossBase import LossBase
    from utils.pt.losses.cgan import CGANLossBase
except Exception as e:
    print(e)
    assert False


class Loss(LossBase):
    pass
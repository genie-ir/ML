try:
    from utils.pt.lossBase import LossBase
    from utils.pt.losses.cgan import CGANLossBase

    class Loss(LossBase):
        pass
except Exception as e:
    print(e)
    assert False



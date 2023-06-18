from utils.pt.lossBase import LossBase

class Loss(LossBase):
    pass

class OHLoss(LossBase):
    def lossfn(self, Yi, Yip1, yi, yip1):
        print('*******************', Yi.shape, Yip1.shape, yi.shape, yip1.shape)
        li = self.criterion(yi, Yi)
        lip1 = self.criterion(yip1, Yip1)

        print(li, lip1)
        assert False
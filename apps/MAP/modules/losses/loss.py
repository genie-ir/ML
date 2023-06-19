from utils.pt.lossBase import LossBase

class Loss(LossBase):
    pass

class OHLoss(LossBase):
    def lossfn(self, Yi, Yip1, yi, yip1):
        loss1, loss2 = self.criterion(yi, Yi) , self.criterion(yip1, Yip1)
        loss = loss1 + loss2
        log = {
            'loss': loss.clone().detach().mean(),
        }
        return loss, log
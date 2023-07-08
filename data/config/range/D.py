from utils.pt.datasets.D import D_Base

class D(D_Base):
    def start(self):
        print('@@@@@@@@@@@@@@@@@@@@@@', self.kwargs)
        assert False
        self.set_length()
    
    def __getitem__(self, i):
        return {

        }

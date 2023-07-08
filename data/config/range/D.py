from utils.pt.datasets.D import D_Base

class D(D_Base):
    def start(self):
        self.set_length(self.config.get('range', 1e3))
    
    def __getitem__(self, i):
        return {
            'index': i
        }

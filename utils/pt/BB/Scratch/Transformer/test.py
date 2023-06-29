import torch
from torch import nn
from utils.pt.building_block import BB

class Test(BB):
    def start(self):
        self.input_nc = self.kwargs.get('input_nc', 3)

    def forward(self, input):
        return self.main(input) 
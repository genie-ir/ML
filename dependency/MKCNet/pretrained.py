import torch
from dependency.MKCNet.main import pretrain

softmax = torch.nn.Softmax(dim=1)


tasknet = pretrain('/content/drive/MyDrive/storage/dr_classifire/best_model.pth')
output, output_M, output_IQ = tasknet(image.cuda())
softmax(output)
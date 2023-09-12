import torch

softmax = torch.nn.Softmax(dim=1)
def pretrain(ckpt):
    model.load_state_dict(torch.load(ckpt))
    return model



tasknet = pretrain('/content/drive/MyDrive/storage/dr_classifire/best_model.pth')
output, output_M, output_IQ = tasknet(image.cuda())
softmax(output)
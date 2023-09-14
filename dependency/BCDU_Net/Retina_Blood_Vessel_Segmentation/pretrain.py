import torch
import numpy as np
from einops import rearrange
from .models import BCDU_net_D3
from .pre_processing import my_PreProc
from .extract_patches import recompone

def pretrain(ckpt):
    model = BCDU_net_D3(input_size = (64,64,1))
    model.load_weights(ckpt)
    
    def Model(img):
        print('A @@@@@@@@@@@', img.shape, img.dtype)
        img = my_PreProc(img)
        print('B @@@@@@@@@@@', img.shape, img.dtype)
        # img = paint_border(img, 64, 64)
        # print('C @@@@@@@@@@@', img.shape, img.dtype)
        # img = rearrange(img, 'b c h w -> b h w c')
        # print('D @@@@@@@@@@@', img.shape, img.dtype)
        img = extract_ordered(img, 64, 64)
        print('E @@@@@@@@@@@', img.shape, img.dtype)
        predictions = model.predict(img, batch_size=2, verbose=1)
        print('F @@@@@@@@@@@', predictions.shape, predictions.dtype)
        predictions = rearrange(predictions, 'b h w c -> b c h w')
        print('G @@@@@@@@@@@', predictions.shape, predictions.dtype)
        # pred_imgs = recompone(predictions,13,12) 
        predictions = recompone(predictions,4,4) 
        print('H @@@@@@@@@@@', predictions.shape, predictions.dtype)
        return None

    return Model




#Divide all the full_imgs in pacthes
def extract_ordered(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    N_patches_h = int(img_h/patch_h) #round to lowest int
    if (img_h%patch_h != 0):
        print ("warning: " +str(N_patches_h) +" patches in height, with about " +str(img_h%patch_h) +" pixels left over")
    N_patches_w = int(img_w/patch_w) #round to lowest int
    if (img_h%patch_h != 0):
        print ("warning: " +str(N_patches_w) +" patches in width, with about " +str(img_w%patch_w) +" pixels left over")
    print ("number of patches per image: " +str(N_patches_h*N_patches_w))
    N_patches_tot = (N_patches_h*N_patches_w)*full_imgs.shape[0]
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))

    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i,:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches






def paint_border(data,patch_h,patch_w):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    img_h=data.shape[2]
    img_w=data.shape[3]
    new_img_h = 0
    new_img_w = 0
    if (img_h%patch_h)==0:
        new_img_h = img_h
    else:
        new_img_h = ((int(img_h)/int(patch_h))+1)*patch_h
    if (img_w%patch_w)==0:
        new_img_w = img_w
    else:
        new_img_w = ((int(img_w)/int(patch_w))+1)*patch_w
    new_data = np.zeros((data.shape[0],data.shape[1],new_img_h,new_img_w))
    new_data[:,:,0:img_h,0:img_w] = data[:,:,:,:]
    return new_data
import cv2, os
import numpy as np
from .AuxFuncs import cvShow, mask_generator
np.seterr(divide='ignore', invalid='ignore')

def patchsize(img, patch_nums):
    H, W  =img.shape
    blockR = H//patch_nums + int(H % patch_nums != 0)
    blockC = W//patch_nums + int(W % patch_nums != 0)
    H, W = blockR*patch_nums, blockC*patch_nums
    return H, W, blockR, blockC

def patchsize_matlab(img, patch_nums):
    # return 1000, 1000, 200, 200
    return 256, 256, 64, 64

def LCN(img, patch_nums):
    # luminosity and contrast normalization
    H, W, blockR, blockC = patchsize_matlab(img, patch_nums)
    mask = mask_generator(img)
    resizeImg = (W, H)
    mask = cv2.resize(mask+0,resizeImg,interpolation=cv2.INTER_NEAREST)
    reImg = cv2.resize(img/255,resizeImg,interpolation=cv2.INTER_NEAREST)
    imgR, imgC = reImg.shape
    M, N = int(imgR / blockR), int(imgC / blockC)
    stdSub, meanSub = np.zeros((M, N)), np.zeros((M, N))
    for i in range(M):  # Row
        for j in range(N):  # Col
            topLeftR, topLeftC = blockR * i + 1, blockC * j + 1
            tempH, tempW = blockR * (i + 1), blockC * (j + 1)
            temp = reImg[topLeftR:tempH, topLeftC:tempW]
            meanSub[i, j] = np.mean(temp) # 均值图像
            stdSub[i, j] = np.std(temp) # 标准差图像
    meanSub[np.isnan(meanSub)]=0
    meanSub[np.isinf(meanSub)]=0
    stdSub[np.isnan(stdSub)]=0
    stdSub[np.isinf(stdSub)]=0
    meanFull = cv2.resize(meanSub, resizeImg, interpolation=cv2.INTER_CUBIC)
    stdFull = cv2.resize(stdSub, resizeImg, interpolation=cv2.INTER_CUBIC)
    meanFull, stdFull = meanFull*mask, stdFull*mask
    mahDist = np.divide(np.subtract(reImg, meanFull), stdFull)
    mahDist = mahDist*mask
    background = np.abs(mahDist)
    background[background < 1] = 1
    background[background != 1] = 0
    meanSub2, stdSub2 = np.zeros((M,N)), np.zeros((M,N))

    for i in range(M):  # Row
        for j in range(N):  # Col
            topLeftR2, topLeftC2 = blockR * i + 1, blockC * j + 1
            tempH2, tempW2 = blockR * (i + 1), blockC * (j + 1)
            temp2 = reImg[topLeftR2:tempH2, topLeftC2:tempW2]
            temp3 = np.ndarray.flatten(temp2)
            patch_back = background[topLeftR2:tempH2, topLeftC2:tempW2]
            patch_back = np.nonzero(np.ravel(patch_back))
            meanSub2[i, j] = np.mean(temp3[patch_back])
            stdSub2[i, j] = np.std(temp3[patch_back])
    meanSub2[np.isnan(meanSub2)]=0
    meanSub2[np.isinf(meanSub2)]=0
    stdSub2[np.isnan(stdSub2)]=0
    stdSub2[np.isinf(stdSub2)]=0
    meanFull2 = cv2.resize(meanSub2, resizeImg, interpolation=cv2.INTER_CUBIC)
    stdFull2 = cv2.resize(stdSub2, resizeImg, interpolation=cv2.INTER_CUBIC)
    meanFull2, stdFull2 = meanFull2*mask, stdFull2*mask
    normalized = np.divide(np.subtract(reImg, meanFull2), stdFull2)
    normalized[np.isnan(normalized)]=0
    normalized[np.isinf(normalized)]=0
    return normalized

if __name__ == '__main__':
    imgpath = './Images/IDRiD_27.jpg'
    img = cv2.imread(imgpath)
    img_g = img[:,:,1]
    result = LCN(img_g, 10)
    cvShow(result)

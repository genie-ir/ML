import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import data
from skimage import filters
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage import io, morphology
from skimage.morphology import disk
from skimage.filters.rank import entropy
from sklearn.metrics import confusion_matrix

#Template creation
template = np.ones((95,95), dtype="uint8") * 0
template = cv.circle(template, (47,47), 46, 255, -1)

def getImg(full_path):
    image = cv.imread(full_path, -1)
    return image

def imgResize(img):
    h = img.shape[0]
    w = img.shape[1]
    perc = 500/w
    w1 = 500
    h1 = int(h*perc)
    img_rs = cv.resize(img,(w1,h1))
    return img_rs

def cannyEdges(img, th1, th2):
    edges = cv.Canny(img, th1, th2)
    return edges

def kmeansclust(img, k):
    img_rsp = img.reshape((-1,1))
    img_rsp = img_rsp.astype('float32')
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 400, 0.99)
    _, labels, (centers) = cv.kmeans(img_rsp, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centers = centers.astype('uint8')
    labels = labels.flatten()
    seg_img = centers[labels.flatten()]
    seg_img = seg_img.reshape(img.shape)
    return seg_img

#Optic Disc Localization
def Optic_Disc_Localization(img_adr):
    assert isinstance(img_adr, str), 'img_adrs={} | it must be str'.format(img_adr)
    img = getImg(img_adr)
    img_rs = imgResize(img)
    img_rc,_,_ = cv.split(img_rs)
    _,img_gc,_ = cv.split(img_rs)
    _,_,img_bc = cv.split(img_rs)
    img_grey = cv.cvtColor(img_rs, cv.COLOR_BGR2GRAY)

    img_k = kmeansclust(img_grey, 7)
    temp = template # cv.imread(template_path, -1)

    #TEMPLATE MATCHING
    metd = cv.TM_CCOEFF_NORMED
    temp_mat = cv.matchTemplate(img_k, temp, metd)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(temp_mat)
    x = max_loc[0]+45
    y = max_loc[1]+45

    temp_mat = np.zeros(img_rc.shape) #img_rc.copy()
    temp_mat[temp_mat>0] = 1
    temp_mat[temp_mat==0] = 1
    img_mark = cv.circle(temp_mat, (x, y) ,40, 255, -1)

    plt.figure()
    plt.imshow(np.hstack([temp_mat, img_gc]), cmap = 'gray', interpolation = 'bicubic')
    plt.savefig(os.path.join('/content/test', os.path.split(img_adr)[1]))
    return img_mark

# if __name__ == '__main__':
#     import pretty_errors
#     from argparse import ArgumentParser
#     parser = ArgumentParser()
#     parser.add_argument(
#         '--fpath',
#         type=str,
#         default=None,
#     )
#     opt, unknown = parser.parse_known_args()
#     Optic_Disc_Localization(opt.fpath)
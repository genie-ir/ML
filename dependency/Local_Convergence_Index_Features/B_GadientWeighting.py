import cv2
import numpy as np
from .AuxFuncs import cvShow
from .A_Normalization import LCN
from skimage import measure

def Gauss_Derivative_Kernel(sig, theta): #高斯偏导核 顺时针移动 theta
    def angle(x,y): # 点的夹角
        if x==0:
            if y > 0: return np.pi/2
            elif y == 0: return 0
            else: return -np.pi/2
        elif x > 0: return np.arctan(y/x)
        else: return np.arctan(y/x) + np.pi
    def GaussDx(point,sig):# x偏导
        x,y = point
        return (np.exp(-(pow(x,2)+pow(y,2))/pow(sig,2)/2) * x) / (np.sqrt(2*np.pi) * pow(sig,3))
    def GaussDy(point,sig): # y偏导
        x,y = point
        return (np.exp(-(pow(x,2)+pow(y,2))/pow(sig,2)/2) * y) / (np.sqrt(2*np.pi) * pow(sig,3))
    rangex = range(np.int(-2 * sig), np.int(2 * sig) + 1) # 3 sigma 原则，发现 3*sig 太大，改为 2*sig
    rangey = range(np.int(-2 * sig), np.int(2 * sig) + 1)
    r = lambda x,y: np.sqrt(pow(x,2)+pow(y,2))
    aft = lambda x,y,the: (r(x,y)*np.cos(angle(x,y) - the), r(x,y)*np.sin(angle(x,y) - the))
    KerDx = np.array([[GaussDx(aft(x,y,theta),sig) for y in rangey] for x in rangex])
    KerDy = np.array([[GaussDy(aft(x,y,theta),sig) for y in rangey] for x in rangex])
    # return KerDx/np.sum(np.abs(KerDx)), KerDy/np.sum(np.abs(KerDy))
    return KerDx, KerDy

def Iteration_Threshold_With_Gauss_Derivative_Conv(img): #此处img非int8型
    H, W = img.shape
    img = np.array(img)
    angels = [np.pi/24*i for i in range(13)]
    M = np.zeros((H, W))
    for sig in range(1,6):
        Iwo = np.zeros((H, W))
        for theta in angels:
            KerDx, KerDy = Gauss_Derivative_Kernel(sig, theta)
            Im = cv2.filter2D(img,-1,KerDx)**2 + cv2.filter2D(img,-1,KerDy)**2
            Iwo += (1-Im)/(1+Im)
        M_sig = np.zeros((H, W))
        for t in np.arange(0.1,1.05,0.05):
            M_sig_t = np.zeros((H, W))
            M_sig_t[Iwo < t] = 1
            labels = measure.label(M_sig_t,connectivity=2)
            properties  = measure.regionprops(labels)
            valid_label = set()
            for prop in properties:
                if prop.area<300 and prop.eccentricity < 0.9\
                and prop.euler_number <= 0 and prop.extent < 0.3:
                    valid_label.add(prop.label)
            M_sig_t_filtered = np.isin(labels, list(valid_label))
            M_sig += M_sig_t_filtered
        M_sig = M_sig > 0
        M += M_sig
    MA_labels = measure.label(M,connectivity=2)
    return MA_labels

kerDx, kerDy = Gauss_Derivative_Kernel(3, np.pi/3)

def main(imgpath):
    img = cv2.imread(imgpath)
    normalized = LCN(img[:,:,1], 5)
    MA_labels = Iteration_Threshold_With_Gauss_Derivative_Conv(normalized)
    return MA_labels
    cvShow(MA_labels>0)

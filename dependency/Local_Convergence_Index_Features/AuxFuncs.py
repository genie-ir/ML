import cv2, os
import numpy as np
import matplotlib.pyplot as plt

def mask_generator(gray):
    ret,threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#OTSU二值化
    contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(threshold, contours, -1, 1, -1)
    for contour in contours:
        cv2.fillConvexPoly(threshold, contour, 1)
        cv2.fillPoly(threshold, contour, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel,iterations=2)
    h, w = gray.shape[:2]
    seedPoint = (int(h/2), int(w/2))
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(closed, mask,seedPoint, 1)
    return closed>0

def retinal_mask_generae(imgdir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    for name in os.listdir(imgdir):
        path = os.path.join(imgdir, name)
        if os.path.isfile(path):
            img = cv2.imread(path, 0)
            mask = mask_generator(img)
            cv2.imwrite(os.path.join(mask_dir, name), mask*255)

def cvShow(img, title = None, max_h = 950, max_w = 1800):
    if img.dtype == 'bool':
        img = bool2bw(img)
    else:img = np.array(img)
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    H,W = img.shape[0],img.shape[1]
    if H > max_h:
        ratio = max_h / H
        H, W = int(ratio * H), int(ratio * W)
    if W > max_w :
        ratio = max_w / W
        H, W = int(ratio * H), int(ratio * W)
    cv2.resizeWindow(title, W, H)
    cv2.imshow(title, img)  # 自动适应图片大小的，不能缩放
    key = cv2.waitKey(0)
    if key == ord('s'):  # wait for key to write or exit
        if title == None:title = 'unNamed'
        cv2.imwrite(title+'.jpg', img)
    cv2.destroyAllWindows()

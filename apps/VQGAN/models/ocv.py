import os, sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


def affine(img, M, **kwargs):
    """M is affine_matrix."""
    h, w = img.shape[:2]
    print(h, w, M, img.shape)
    return cv2.warpAffine(img, np.float32(M), (w, h))

def translation(img, tx, ty, **kwargs):
    """
        Example: tx=100, ty=-50
        Negative values of tx will shift the image to the left
        Positive values will shift the image to the right
        Negative values of ty will shift the image up
        Positive values will shift the image down
    """
    M = np.float32([
        [1, 0, tx],
        [0, 1, ty],
    ])
    if kwargs.get('return_M', False): # OPTIONAL
        return M
    return affine(img, M)

def rotation(img, theta, s=1, **kwargs):
    """
        Example: theta=60, s=1
        if theta is positive, our output image will rotate counterclockwise(<-). Similarly, 
        if theta is negative the image will rotate clockwise(->).
        * s = 1 -> rotated image will have the same dimensions. 
        * s = 2 -> rotated image will have the doubled in size.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(kwargs.get('era_center', center), theta, float(s))
    if kwargs.get('return_M', False): # OPTIONAL
        return M
    return affine(img, M)

def ROT(img, **kwargs):
    """
        params: theta, tx, ty
    """
    kwargs['theta'] = int(kwargs['theta'].round().item())
    kwargs['tx'] = int(kwargs['tx'].round().item())
    kwargs['ty'] = int(kwargs['ty'].round().item())
    
    print('222222222222222222', kwargs['theta'], kwargs['tx'], kwargs['ty'])
    assert False

    R = rotation(img, **kwargs, return_M=True)
    T = translation(img, **kwargs, return_M=True)
    T[:, :-1] = 0
    M = R + T
    return affine(img, M)

def flip(img, mode='both', **kwargs):
    """
        A value 1 indicates that we are going to flip our image around the y-axis (horizontal flipping). 
        On the other hand, a value 0 indicates that we are going to flip the image around the
        x-axis (vertical flipping). 
        If we want to flip the image around both axes, we will use a negative value (e.g. -1).
    """
    if mode == 'both' or mode == 'vh' or mode == 'hv':
        mode = -1
    elif mode == 'v' or mode == 'vertical':
        mode = 0
    elif mode == 'h' or mode == 'horizontal':
        mode = 1
    else:
        assert False
    return cv2.flip(img, mode)

def resize(img, **kwargs):
    H, W = img.shape[:2] # NOTE: name ordering is a rule!
    h = int(kwargs.get('h', -1)) # OPTIONAL
    w = int(kwargs.get('w', -1)) # OPTIONAL
    
    aspect_ratio = True
    if w != -1 and h != -1:
        aspect_ratio = False

    if aspect_ratio:
        if w != -1:
            ratio = float(w) / W
            h = int(H * ratio)
        elif h != -1:
            ratio = float(h) / H
            w = int(W * ratio)
        else:
            assert False, 'At least one of w or h, or both of them, must be provided.'
    return cv2.resize(img, (w, h))

from os.path import join as ospjoin, split as ospsplit, dirname as ospdirname

def makedirs(fpath, **kwargs):
    upstream_path = str(kwargs.get('_file_', None))
    fpath0, fpath1 = ospsplit(fpath)
    if upstream_path != None:
        fpath0 = ospjoin(ospsplit(upstream_path)[0], fpath0)
    fpath0 = fpath0.replace('.', '')
    os.makedirs(fpath0, exist_ok=True)
    return ospjoin(fpath0, fpath1)

def save(fpath, img):
    return cv2.imwrite(makedirs(fpath, _file_=__file__), img)

def load(fpath, mode='color'):
    """Loading our image with a cv2.imread() function, this function loads the image in BGR order"""
    if mode == 'color':
        mode = cv2.IMREAD_COLOR # There’s also another option for loading a color image: we can just put the number 1 instead cv2.IMREAD_COLOR and we will obtain the same output.
    elif mode == 'gray':
        mode = cv2.IMREAD_GRAYSCALE # The value that’s needed for loading a grayscale image is cv2.IMREAD_GRAYSCALE, or we can just put the number 0 instead as an argument.
    else:
        assert False
    img = cv2.imread(fpath, mode)
    return img

def split(img):
    """We can split the our image into 3 three channels (b, g, r) for normal color image, otherwise this functin split image into their channells."""
    return cv2.split(img) # return 3 channells: b, g, r seperaitly, if img was a normal color image; otherwise img splits to their own channells and those channells seperaily will return.

def convert_color(img, mode='bgr2rgb'):
    """
        Example:
            b, g, r = split(img) # img is a normal color image loaded by cv2.
            rgb_version_img = cv2.merge([r, g, b])
            * This tequnice can be used for any multi channell image.
    """
    if mode == 'bgr2rgb':
        mode = cv2.COLOR_BGR2RGB
    else:
        assert False
    return cv2.cvtColor(img, mode)

def show(img, mode='cv2'):
    if mode == 'cv2':
        cv2_imshow(img)
    elif mode == 'plt':
        plt.imshow(convert_color(img))
        plt.axis('off')
        plt.show()
    else:
        assert False
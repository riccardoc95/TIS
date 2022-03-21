import pandas as pd
from astropy.convolution import Gaussian2DKernel
from scipy import ndimage
from skimage.restoration import denoise_tv_bregman
from skimage.restoration import denoise_wavelet
import numpy as np

def gaussian(img, sigma=2, x_size=None, y_size=None):
    if x_size is None or y_size is None:
        if sigma < 1/4:
            x_size=3
            y_size=3
        else:
            x_size = int(8*sigma + 1)
            y_size = int(8*sigma + 1)
    
    #x_size=3
    #y_size=3
    
    kernel = Gaussian2DKernel(sigma, x_size=x_size, y_size=y_size)
    kernel.normalize()
    kernel_array = kernel.array
    data = ndimage.convolve(img.astype(float), kernel_array, mode='reflect', cval=0)
    return data

def anisodiff(img, niter = 30, gamma = 0.14, kappa = 15):

    # initial condition
    data = img

    # center pixel distances
    dx = 1
    dy = 1
    dd = np.sqrt(2)

    # 2D finite difference windows
    windows = [
        np.array(
                [[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64
        ),
        np.array(
                [[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64
        ),
    ]

    for r in range(niter):
        # approximate gradients
        nabla = [ ndimage.filters.convolve(data, w) for w in windows ]

        # approximate diffusion function
        diff = [ 1./(1 + (n/kappa)**2) for n in nabla]
        
        # update image
        terms = [diff[i]*nabla[i] for i in range(4)]
        terms += [(1/(dd**2))*diff[i]*nabla[i] for i in range(4, 8)]
        data = data + gamma*(sum(terms))

        return data
    
def total_variation(img, weight=120):
    data = denoise_tv_bregman(img.astype(float), weight=weight, max_iter=500, eps=0.0001, isotropic=True)
    return data

def wavelet(img, sigma=None):
    data = denoise_wavelet(img, sigma=sigma, wavelet='bior1.3')
    return data
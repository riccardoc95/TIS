import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image
from scipy.optimize import fmin


# Mask RMS
def generate_mask(data, thresholds, d=5, distance=False):
    data2 = data > thresholds
    data2 = ndimage.binary_dilation(data2, structure=ndimage.generate_binary_structure(2,1), iterations=d)
    if distance:
        distance = ndimage.distance_transform_edt(data2)
        return distance
    else:
        return data2
    
    
# Segmentation Mask
##TODO: Sistemare moments!!!!
def convex_mask(mask):
    return convex_hull_image(mask)


def ellipse(h, k, a, b, phi, x, y):
    xp = (x-h)*np.cos(phi) + (y-k)*np.sin(phi)
    yp = -(x-h)*np.sin(phi) + (y-k)*np.cos(phi)
    return (xp/(a+1e-05))**2 + (yp/(b+1e-05))**2 <= 1
  
def func(theta, x,y, mask):
    (h,k,a,b,phi) = theta
    return (np.logical_xor(ellipse(h,k,a,b,phi,x,y), mask)).sum()

def moments(mask, x, y):
    N = float(mask.sum())
    h0 = (x*mask).sum() / N
    k0 = (y*mask).sum() / N
    sxx = (((x-h0)**2)*mask).sum() / N
    syy = (((y-k0)**2)*mask).sum() / N
    sxy = (((x-h0)*(y-k0))*mask).sum() / N

    term1 = 2*(sxx + syy)
    term2 = np.sign(sxy) * np.sqrt(4*(sxx-syy)**2 + 16*sxy*sxy)

    asq = term1 + term2
    bsq = term1 - term2
    diff = asq - bsq
    if diff == 0:
        phi = 0.0
    else:
        if 8*sxy / diff > 1:
            phi = 0.5*np.arcsin(1)
        elif 8*sxy / diff < -1:
            phi = 0.5*np.arcsin(-1)
        else:
            phi = 0.5*np.arcsin(8*sxy / diff)

    return h0, k0, np.sqrt(asq), np.sqrt(bsq), phi

def update_mask(mask, mask0, x, y):
    h0, k0, a0, b0, phi0 = moments(mask, x, y)
    return mask0 * ellipse(h0, k0, a0, b0, phi0, x, y)
 
def ellipse_mask(mask0, iter=8):
    x, y = np.indices(mask0.shape)
    mask = mask0
    for k in range(iter):
        mask = update_mask(mask, mask0, x, y)
    h0, k0, a0, b0, phi0 = moments(mask, x, y)
    #theta = (h0, k0, a0, b0, phi0)
    #hopt, kopt, aopt, bopt, phiopt = fmin(func, theta, args=(x, y, mask), disp=False, maxiter=10)
    hopt, kopt, aopt, bopt, phiopt = h0, k0, a0, b0, phi0
    return ellipse(hopt, kopt, aopt, bopt, phiopt, x, y)
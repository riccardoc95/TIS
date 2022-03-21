import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval


norm = ZScaleInterval()

def open_fits(path):
    return fits.getdata(path)

def write_fits(obj, path):
    hdu = fits.PrimaryHDU(obj)
    hdu.writeto(path)

def minmaxscaler(x, min_t=0, max_t=1):
    y = x[~np.isnan(x)]
    return (x - y.min())/(y.max()-y.min()) * (max_t - min_t) + min_t

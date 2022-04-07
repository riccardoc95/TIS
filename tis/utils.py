import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt


norm = ZScaleInterval()


def open_fits(path):
    return fits.getdata(path)


def write_fits(obj, path):
    hdu = fits.PrimaryHDU(obj)
    hdu.writeto(path)


def write_info(info, path):
    info.to_csv(path, index=False)


def minmaxscaler(x, min_t=0, max_t=1):
    y = x[~np.isnan(x)]
    return (x - y.min())/(y.max()-y.min()) * (max_t - min_t) + min_t


# PLOTS
def plot_centers(img, centers, true=None, idxs=None, save_path=None):
    if idxs is None:
        idxs = np.arange(0, centers.shape[0])
       
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(norm(img), cmap='gray', interpolation='nearest')
    for idx in idxs[:]:
        x, y = centers[idx]
        if true is None:
            plt.scatter(y, x, 5, 'red')
        else:
            if true[x,y] != 0:
                plt.scatter(y, x, 5, 'navy')
            else:
                plt.scatter(y, x, 5, 'red')
    if save_path is None: 
        plt.show()
    else:
        fig.savefig(save_path, dpi=fig.dpi)
        
        
def plot_segmentation(segm, save_path=None):
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(segm, interpolation='nearest')
    if save_path is None: 
        plt.show()
    else:
        fig.savefig(save_path, dpi=fig.dpi)

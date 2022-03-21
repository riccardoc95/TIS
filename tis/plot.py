import numpy as np
import matplotlib.pyplot as plt
from utils import norm


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
    #plt.imshow(segm, cmap='gray', interpolation='nearest')
    plt.imshow(segm, interpolation='nearest')
    if save_path is None: 
        plt.show()
    else:
        fig.savefig(save_path, dpi=fig.dpi)
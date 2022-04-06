import numpy as np

from denoise import gaussian, anisodiff, total_variation
from thresholds import thresholds
from persistent_diagrams import persistent_diagrams
from filters import kmeans
from mask import generate_mask
from utils import norm, minmaxscaler

import gc
import warnings


warnings.filterwarnings("ignore")


def find_sigma(img, rms, t=2, d=5, sigma_start=1.25, sigma_eps=0.1, patience=0):
    sigma_star = sigma_start
    sigma_star_next = sigma_star
    sigma_step = sigma_start / 2
    patience = patience
    
    init = True
    while sigma_step >= sigma_eps:
    
        sigmas = np.arange(sigma_start,0,-sigma_step)

        for k, sigma in enumerate(sigmas):
            data = gaussian(img, sigma=sigma)
            thrs = thresholds(img, rms, t=t)
            mask = generate_mask(data, thrs, d=d, distance=False)
    
            im = minmaxscaler(data) * mask

            dgm, centers, ends, lifetime = persistent_diagrams(im, plot=False)

            idxs = kmeans(lifetime)
            n = idxs.size
            
            if init:
                n_old = n
                init = False
            else:
                if n > 2 * n_old:
                    if patience > 0:
                        init=True
                        patience -= 1
                    break
                else:
                    n_old = n
                    sigma_star = sigma_star_next
                    sigma_star_next = sigma
            
        
        # Aggiornamento sigma e step
        if n > 2 * n_old:
            sigma_start = sigma + sigma_step
        else:
            sigma_start = sigma
        sigma_step = sigma_step/2
        sigma_start = sigma_start - sigma_step
    
    # Clean memory
    del data
    del mask
    del im
    del dgm
    del centers
    del ends
    del lifetime
    del idxs
    gc.collect()
    
    return sigma_star


def preprocess(img, rms, sigma=1.25, t=2, d=5):
    if sigma is None:
        sigma = find_sigma(img, rms, t=t, d=d, sigma_start=1.25, sigma_eps=0.1, patience=0)
    data = gaussian(img, sigma=sigma)
    thrs = thresholds(img, rms, t=t)
    mask = generate_mask(data, thrs, d=d, distance=False)
    
    im = minmaxscaler(data) * mask
    return im
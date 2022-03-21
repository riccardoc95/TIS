import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from .denoise import gaussian, anisodiff, total_variation
from .thresholds import thresholds
from .mask import generate_mask, convex_mask, ellipse_mask
from .persistent_diagrams import persistent_diagrams
from .filters import kmeans
from .filters import kmeans, mean_shift, spectral, agglomerative
from .filters import ps_entropy, gaussian_mixture, bayesian_gaussian_mixture
from .utils import norm, minmaxscaler


def find_sigma(img, rms, t=2, d=3, sigma_start=1.25, sigma_eps=0.1, patience=0):
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


def find_centers(img, rms, sigma=None, t=2, d=3):
    if sigma is None:
        sigma = find_sigma(img, rms, t=t, d=d, sigma_start=1.25, sigma_eps=0.1, patience=0)
    data = gaussian(img, sigma=sigma)
    thrs = thresholds(img, rms, t=t)
    mask = generate_mask(data, thrs, d=d, distance=False)
    
    im = minmaxscaler(data) * mask
    
    dgm, centers, ends, lifetime = persistent_diagrams(im, plot=False)
    
    idxs = kmeans(lifetime)
    idxs = np.flip(idxs[(np.argsort(dgm[idxs,1]))])
    
    return centers[idxs]


def segmentation(img, rms, sigma=None, t=2, d=3, convex=False, ellipse=False, info=False, verbose=True):
    if sigma is None:
        sigma = find_sigma(img, rms, t=t, d=d, sigma_start=1.25, sigma_eps=0.1, patience=0)
    data = gaussian(img, sigma=sigma)
    thrs = thresholds(img, rms, t=t)
    mask = generate_mask(data, thrs, d=d, distance=False)
    
    im = minmaxscaler(data) * mask
    
    dgm, centers, ends, lifetime = persistent_diagrams(im, plot=False)
    
    idxs = kmeans(lifetime)
    idxs = np.flip(idxs[(np.argsort(dgm[idxs,1]))])
    
    info_df = pd.DataFrame({'id':[i+1 for i in range(idxs.size)],
                            'x':[centers[i,0] for i in idxs],
                            'y':[centers[i,1] for i in idxs],
                            'birth':[dgm[i,0] for i in idxs],
                            'death':[dgm[i,1] for i in idxs],
                            'lifetime':[lifetime[i] for i in idxs],
                            'area': np.nan,
                            'flusso': np.nan,
                            'errore': np.nan,
                            'x_min': np.nan,
                            'y_min': np.nan,
                            'x_max': np.nan,
                            'y_max': np.nan})
    img_components = np.zeros_like(im)
    
    
    if verbose:
        iterator = tqdm(idxs)
    else:
        iterator = idxs
        
    for idx in iterator:
        x,y = centers[idx]
        
        # Se il punto è fuori l'immagine, continua
        if (x > im.shape[0] - 1) or (y > im.shape[1] - 1):
            continue
            
        id_ = info_df[(info_df['x'] == x) & (info_df['y'] == y)]['id'].values[0] - 1

        img_temp = np.logical_and(-im <0.0 * dgm[idx,0] + 1*dgm[idx, 1], -im >= dgm[idx, 0])

        component_at_idx = ndimage.label((img_temp))[0]

        del(img_temp)
        
        component = component_at_idx == component_at_idx[x,y]
        
        # Se l'oggetto non c'è o è grande come tutta l'immagine, continua
        if (component.sum() == 0) or (component.sum() == im.size):
            continue
        
        # Modifica forma della segmentazione
        if ellipse:
            component = ellipse_mask(component)
        elif convex:
            component = convex_mask(component)
        
        # INFO
        info_df.loc[id_,'area'] = component.sum()
        info_df.loc[id_,'flusso'] = img[component].sum()
        info_df.loc[id_,'errore'] = (rms[component]**2).sum()
        
        x_t, y_t = np.where(component)
        info_df.loc[id_,'x_min'] = x_t.min()
        info_df.loc[id_,'y_min'] = y_t.min()
        info_df.loc[id_,'x_max'] = x_t.max()
        info_df.loc[id_,'y_max'] = y_t.max()

        img_components[component] = (1 + id_)

    if info:
        
        return img_components, info_df
    else:
        return img_components
    
    
def image_segmentation(img, rms, patch_size=1000, sigma=None, t=2, d=3, convex=False, ellipse=False, info=False, verbose=True):
    ##TODO: stimare sigma per ogni patch
    img_segm = np.zeros_like(img)
    img_info = pd.DataFrame({'x':[], 'y':[], 'birth':[], 'death':[], 'lifetime':[], 'area': [], 'flusso': [], 'errore': []})
    n_obj = 0
    
    if verbose:
        pbar = tqdm(total = int(img.shape[0]/patch_size) * int(img.shape[1]/patch_size), bar_format = "{desc}: {percentage:.2f}%|{bar}| [{elapsed}<{remaining}]")
    for i in range(0,img.shape[0], patch_size):
        for j in range(0,img.shape[1], patch_size):
            if info:
                segm, inf0 = segmentation(img[i:i+patch_size, j:j+patch_size], rms[i:i+patch_size, j:j+patch_size], sigma=sigma, t=t, d=d, convex=convex, ellipse=ellipse, info=info, verbose=False)
            else:
                segm = segmentation(img[i:i+patch_size, j:j+patch_size], rms[i:i+patch_size, j:j+patch_size], sigma=sigma, t=t, d=d, convex=convex, ellipse=ellipse, info=info, verbose=False)
            if inf0.loc[0,'area'] == np.nan:
                continue
            else:
                segm[segm > 0] = segm[segm > 0] + n_obj
                inf0['id'] = inf0['id'] + n_obj
                inf0['x'] = inf0['x'] + i 
                inf0['y'] = inf0['y'] + j 
                img_segm[i:i+patch_size, j:j+patch_size] = segm
                img_info = pd.concat([img_info, inf0])
                n_obj += inf0.shape[0]
            if verbose:
                pbar.update(1)
    if verbose:
        pbar.close()
    if info:
        return img_segm,img_info
    else:
        return img_segm


# Deprecated
def sigmas_scale_centers(img, rms,sigmas=[1], t=2, d=3):
    glob_centers = np.empty((0,2))
    for sigma in sigmas:
        centers = find_centers(img,rms,sigma=sigma, t=t, d=d)

        glob_centers = np.concatenate((glob_centers, centers), axis=0)
        glob_centers = np.unique(glob_centers, axis=0)


        res1 = np.array(list(set(map(tuple, glob_centers)) - set(map(tuple, centers))))
        print()
        print('----------')
        print("Sigma:", round(sigma,2))
        print("N. centers:", centers.shape[0], flush=True)
        print("N. glob centers:", glob_centers.shape[0], flush=True)
        print("punti persi:",res1.shape[0])
        print('----------')
    return glob_centers
        

    


    
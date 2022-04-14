import numpy as np
from scipy import ndimage

from filters import kmeans
from persistent_diagrams import persistent_diagrams
from mask import convex_mask, ellipse_mask


def segmentation(data, img=None, rms=None, convex=False, ellipse=False, start_id=0):
    dgm, centers, ends, lifetime = persistent_diagrams(data, plot=False)
    
    idxs = kmeans(lifetime)
    idxs = np.flip(idxs[(np.argsort(dgm[idxs,1]))])
    
    info = {'id':[],
            'x': [],
            'y': [],
            'x_min': [],
            'y_min': [],
            'x_max': [],
            'y_max': [],
            'area': [],
            'flusso': [],
            'errore': []}
    data_components = np.zeros_like(data)
    
    
    i = 1
    for idx in idxs:
        x,y = centers[idx]
        
        # Se il punto è fuori l'immagine, continua
        if (x > data.shape[0] - 1) or (y > data.shape[1] - 1):
            continue

        data_temp = np.logical_and(-data <0.0 * dgm[idx,0] + 1*dgm[idx, 1], -data >= dgm[idx, 0])

        component_at_idx = ndimage.label((data_temp))[0]

        del(data_temp)
        
        component = component_at_idx == component_at_idx[x,y]
        
        # Se l'oggetto non c'è o è grande come tutta l'immagine, continua
        if (component.sum() == 0) or (component.sum() == data.size):
            continue
        
        # Modifica forma della segmentazione
        if ellipse:
            component = ellipse_mask(component)
        elif convex:
            component = convex_mask(component)
        
        # INFO
        info['id'].append(start_id + i)
        info['x'].append(x)
        info['y'].append(y)
        x_t, y_t = np.where(component)
        info['x_min'].append(x_t.min())
        info['y_min'].append(y_t.min())
        info['x_max'].append(x_t.max())
        info['y_max'].append(y_t.max())
        info['area'].append(component.sum())
        if img is not None:
            info['flusso'].append(img[component].sum())
        else:
            info['flusso'].append(np.nan)
        if rms is not None:
            info['errore'].append((rms[component]**2).sum())
        else:
            info['errore'].append(np.nan)
        
        data_components[component] = start_id + i
        
        i += 1
        

    return data_components, info

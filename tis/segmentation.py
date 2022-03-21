import numpy as np
import pandas as pd
from scipy import ndimage
from filters import kmeans
from persistent_diagrams import persistent_diagrams
from mask import generate_mask, convex_mask, ellipse_mask

def segmentation(img, sigma=1.25, t=2, d=3, convex=False, ellipse=False, start_id=0):
    dgm, centers, ends, lifetime = persistent_diagrams(img, plot=False)
    
    idxs = kmeans(lifetime)
    idxs = np.flip(idxs[(np.argsort(dgm[idxs,1]))])

    img_components = np.zeros_like(img)
    
    i = 1
    for idx in idxs:
        x,y = centers[idx]
        
        # Se il punto è fuori l'immagine, continua
        if (x > img.shape[0] - 1) or (y > img.shape[1] - 1):
            continue

        img_temp = np.logical_and(-img <0.0 * dgm[idx,0] + 1*dgm[idx, 1], -img >= dgm[idx, 0])

        component_at_idx = ndimage.label((img_temp))[0]

        del(img_temp)
        
        component = component_at_idx == component_at_idx[x,y]
        
        # Se l'oggetto non c'è o è grande come tutta l'immagine, continua
        if (component.sum() == 0) or (component.sum() == img.size):
            continue
        
        # Modifica forma della segmentazione
        if ellipse:
            component = ellipse_mask(component)
        elif convex:
            component = convex_mask(component)
        
        img_components[component] = start_id + i
        i+=1

    return img_components


"""
# INFO
info_df.loc[id_,'area'] = component.sum()
info_df.loc[id_,'flusso'] = img[component].sum()
info_df.loc[id_,'errore'] = (rms[component]**2).sum()

x_t, y_t = np.where(component)
info_df.loc[id_,'x_min'] = x_t.min()
info_df.loc[id_,'y_min'] = y_t.min()
info_df.loc[id_,'x_max'] = x_t.max()
info_df.loc[id_,'y_max'] = y_t.max()


"""

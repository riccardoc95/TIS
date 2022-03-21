import numpy as np
import cripser
import tcripser
from persim import plot_diagrams


def persistent_diagrams(img, plot=False):
    pds = tcripser.computePH(-img, maxdim=0)

    dgm = pds[:,1:3]
    dgm[-1,1] = 0
    centers = pds[:,3:5].astype('int')
    ends = pds[:,6:8].astype('int')
    lifetime = np.abs(dgm[:,1]- dgm[:,0])
    
    if plot:
        plot_diagrams(dgm)
    
    return dgm, centers, ends, lifetime
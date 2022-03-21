import numpy as np
from astropy.stats import SigmaClip
from photutils.background import SExtractorBackground


def thresholds(img, rms, t=2):
    sigma_clip = SigmaClip(sigma=3.0)
    bkg = SExtractorBackground(sigma_clip)
    bkg_value = bkg.calc_background(img)
    if t is None:
        t = np.quantile(((img-bkg_value)/rms),0.98)
    thresholds = bkg_value + (t * rms)
    return thresholds

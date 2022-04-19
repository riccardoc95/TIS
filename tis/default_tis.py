######################
##### ISTRUZIONI #####
######################
#
## DATAMANAGER PARAMETERS
#  - DIM_PATCH = dimensione del ritaglio dell'immagine
#  - DIM_OVERLAP = dimensione dell'overlap
#
## SEGMENTATION PARAMETERS
#  - T = parametro di threshold per la maschera RMS
#  - D = parametro di dilatazione della maschera RMS
#  - CONVEX = forza la segmentazione ad essere convessa
#  - ELLIPSE = forza la segmentazione ad essere un ellisse
#
#  - FILTER_TYPE =
#  - DENOISE_TYPE =
#
#
#  # GAUSSIAN
#  - SIGMA = paramentro per il denoising Gaussiano (se =None viene stimato)
#  - X_SIZE =
#  - Y_SIZE =
#
#  # ANISODIFF
#  - N_ITER =
#  - GAMMA =
#  - KAPPA =
#
#  # TOTAL_VARIATION
#  - WEIGHT =
#  - MAX_ITER =
#  - EPS =
#  - ISOTROPIC =
#
#  # WAVELET
#  - SIGMA_W =
#  - WAVELET =
#
#

[INPUT PARAMETERS]
DATA_FOLDER = ./data/grid/
IMG_FILE = img.fits
RMS_FILE = rms.fits

[OUTPUT PARAMETERS]
OUT_FOLDER = ./out
OUT_FILE = segm.fits
OUT_INFO = segm.csv

[DATAMANAGER PARAMETERS]
DIM_PATCH = 500
DIM_OVERLAP = 50


[SEGMENTATION PARAMETERS]
T = 2
D = 5
CONVEX = False
ELLIPSE = False

#[FILTER AND DENOISE]
FILTER_TYPE = kmeans
DENOISE_TYPE = gaussian

#[GAUSSIAN]
SIGMA = 1.15
X_SIZE = None
Y_SIZE = None

#[ANISODIFF]
N_ITER = 30
GAMMA = 0.14
KAPPA = 15

#[TOTAL_VARIATION]
WEIGHT = 120
MAX_ITER = 500
EPS = 0.0001
ISOTROPIC = True

#[WAVELET]
SIGMA_W = None
WAVELET = bior1.3

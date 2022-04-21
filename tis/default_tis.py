######################
##### ISTRUZIONI #####
######################
#
## DATAMANAGER PARAMETERS
#  - DIM_PATCH = (INTEGER) dimensione del ritaglio dell'immagine
#  - DIM_OVERLAP = (INTEGER) dimensione dell'overlap
#
## SEGMENTATION PARAMETERS
#  - T = (FLOAT) parametro di threshold per la maschera RMS
#  - D = (INTEGER) parametro di dilatazione della maschera RMS
#  - CONVEX = (BOOLEAN) forza la segmentazione ad essere convessa
#  - ELLIPSE = (BOOLEAN) forza la segmentazione ad essere un ellisse
#
#  - FILTER_TYPE = (STRING) tipo di filtro per il diagramma di persistenza, da scegliere tra:
#                  - persistent_entropy,
#                  - gaussian_mixture,
#                  - bayesian_gaussian_mixture,
#                  - kmeans,
#                  - mean_shift,
#                  - spectral,
#                  - agglomerative
#
#  - DENOISE_TYPE = (STRING) tipo di denoise da applicare all'immagine, da scegliere tra:
#                   - gaussian,
#                   - anisodiff,
#                   - total_variation,
#                   - wavelet
#
#
#  # GAUSSIAN
#  - SIGMA = (FLOAT/NONE) paramentro per il denoising Gaussiano (se =None viene stimato)
#  - X_SIZE = (INT)
#  - Y_SIZE = (INT)
#
#  # ANISODIFF
#  - N_ITER = (INT)
#  - GAMMA = (FLOAT)
#  - KAPPA = (INT)
#
#  # TOTAL_VARIATION
#  - WEIGHT = (FLOAT)
#  - MAX_ITER = (INT)
#  - EPS = (FLOAT)
#  - ISOTROPIC = (BOOLEAN)
#
#  # WAVELET
#  - SIGMA_W = (FLOAT)
#  - WAVELET = (STRING)
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

FILTER_TYPE = kmeans
DENOISE_TYPE = gaussian

[GAUSSIAN]
SIGMA = 1.15
X_SIZE = None
Y_SIZE = None

[ANISODIFF]
N_ITER = 30
GAMMA = 0.14
KAPPA = 15

[TOTAL_VARIATION]
WEIGHT = 120
MAX_ITER = 500
EPS = 0.0001
ISOTROPIC = True

[WAVELET]
SIGMA_W = None
WAVELET = bior1.3

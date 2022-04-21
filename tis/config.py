import configparser
import os
import sys
import pywt


# DEFAULT PARAMETERS
input_parameters = {
    "DATA_FOLDER": "./data",
    "IMG_FILE": "img.fits",
    "RMS_FILE": "rms.fits",
}

output_parameters = {
    "OUT_FOLDER": "./out",
    "OUT_FILE": "segm.fits",
    "OUT_INFO": "segm.csv",
}

datamanager_paramenters = {
    "DIM_PATCH": 512,
    "DIM_OVERLAP": 50,
}

segmentation_parameters = {
    "T": 2,
    "D": 3,
    "CONVEX": False,
    "ELLIPSE": False,
    "FILTER_TYPE": "kmeans",
    "DENOISE_TYPE": "gaussian",
}

gaussian_parameters = {
    "SIGMA": None,
    "X_SIZE": None,
    "Y_SIZE": None,
}

anisodiff_paramenters = {
    "N_ITER": 30,
    "GAMMA": 0.14,
    "KAPPA": 15,
}

totalvariation_paramenters = {
    "WEIGHT": 120,
    "MAX_ITER": 500,
    "EPS": 0.0001,
    "ISOTROPIC": True,
}

wavelet_paramenters = {
    "SIGMA_W": None,
    "WAVELET": "bior1.3",
}


# READ PARAMETERS FROM .CONF FILE
parser = configparser.ConfigParser()
cwd = os.getcwd()
config_file_path = os.path.join(cwd, "tis.conf")
parser.read(config_file_path)


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False


for section_name in parser.sections():
    if section_name == "INPUT PARAMETERS":
        for key, value in parser.items(section_name):
            input_parameters[key.upper()] = str(value)
    elif section_name == "OUTPUT PARAMETERS":
        for key, value in parser.items(section_name):
            output_parameters[key.upper()] = str(value)
    elif section_name == "DATAMANAGER PARAMETERS":
        for key, value in parser.items(section_name):
            if is_float(value):
                value = float(value)
            else:
                sys.exit("[tis.conf ERROR]: datamanager parameters are not integers")
            if float(value).is_integer():
                datamanager_paramenters[key.upper()] = int(value)
            else:
                sys.exit("[tis.conf ERROR]: datamanager parameters are not integers")
    elif section_name == "SEGMENTATION PARAMETERS":
        for key, value in parser.items(section_name):
            if key.upper() == 'T':
                if is_float(value):
                    segmentation_parameters[key.upper()] = float(value)
                else:
                    sys.exit("[tis.conf ERROR]: segmentation parameter T is not float")
            elif key.upper() == 'D':
                if is_float(value):
                    value = float(value)
                else:
                    sys.exit("[tis.conf ERROR]: segmentation parameter D is not integer")
                if float(value).is_integer():
                    segmentation_parameters[key.upper()] = int(value)
                else:
                    sys.exit("[tis.conf ERROR]: segmentation parameter D is not integer")
            elif key.upper() == 'CONVEX':
                if value.upper() == 'TRUE':
                    segmentation_parameters[key.upper()] = True
                elif value.upper() == 'FALSE':
                    segmentation_parameters[key.upper()] = False
                else:
                    sys.exit("[tis.conf ERROR]: segmentation parameter CONVEX is not boolean")
            elif key.upper() == 'ELLIPSE':
                if value.upper() == 'TRUE':
                    segmentation_parameters[key.upper()] = True
                elif value.upper() == 'FALSE':
                    segmentation_parameters[key.upper()] = False
                else:
                    sys.exit("[tis.conf ERROR]: segmentation parameter ELLIPSE is not boolean")
            elif key.upper() == 'FILTER_TYPE':
                if value.lower() in ['persistent_entropy',
                                     'gaussian_mixture',
                                     'bayesian_gaussian_mixture',
                                     'kmeans',
                                     'mean_shift',
                                     'spectral',
                                     'agglomerative']:
                    segmentation_parameters[key.upper()] = value.lower()
                else:
                    sys.exit("[tis.conf ERROR]: segmentation parameter FILTER_TYPE is not correct")
            elif key.upper() == 'DENOISE_TYPE':
                if value.lower() in ['gaussian',
                                     'anisodiff',
                                     'total_variation',
                                     'wavelet']:
                    segmentation_parameters[key.upper()] = value.lower()
                else:
                    sys.exit("[tis.conf ERROR]: segmentation parameter DENOISE_TYPE is not correct")
    elif section_name == "GAUSSIAN":
        for key, value in parser.items(section_name):
            if key.upper() == 'SIGMA':
                if value.upper() == 'NONE':
                    gaussian_parameters[key.upper()] = None
                elif is_float(value):
                    gaussian_parameters[key.upper()] = float(value)
                else:
                    sys.exit("[tis.conf ERROR]: gaussian parameter SIGMA is not correct")
            elif key.upper() == 'X_SIZE':
                if value.upper() == 'NONE':
                    gaussian_parameters[key.upper()] = None
                elif is_float(value):
                    if float(value).is_integer():
                        gaussian_parameters[key.upper()] = int(value)
                    else:
                        sys.exit("[tis.conf ERROR]: gaussian parameter X_SIZE is not correct")
                else:
                    sys.exit("[tis.conf ERROR]: gaussian parameter X_SIZE is not correct")
            elif key.upper() == 'Y_SIZE':
                if value.upper() == 'NONE':
                    gaussian_parameters[key.upper()] = None
                elif is_float(value):
                    if float(value).is_integer():
                        gaussian_parameters[key.upper()] = int(value)
                    else:
                        sys.exit("[tis.conf ERROR]: gaussian parameter Y_SIZE is not correct")
                else:
                    sys.exit("[tis.conf ERROR]: gaussian parameter Y_SIZE is not correct")
    elif section_name == "ANISODIFF":
        for key, value in parser.items(section_name):
            if key.upper() == 'N_ITER':
                if is_float(value):
                    if float(value).is_integer():
                        anisodiff_paramenters[key.upper()] = int(value)
                    else:
                        sys.exit("[tis.conf ERROR]: anisodiff parameter N_ITER is not correct")
                else:
                    sys.exit("[tis.conf ERROR]: anisodiff parameter N_ITER is not correct")
            elif key.upper() == 'GAMMA':
                if is_float(value):
                    anisodiff_paramenters[key.upper()] = float(value)
                else:
                    sys.exit("[tis.conf ERROR]: anisodiff parameter GAMMA is not correct")
            elif key.upper() == 'KAPPA':
                if is_float(value):
                    if float(value).is_integer():
                        anisodiff_paramenters[key.upper()] = int(value)
                    else:
                        sys.exit("[tis.conf ERROR]: anisodiff parameter KAPPA is not correct")
                else:
                    sys.exit("[tis.conf ERROR]: anisodiff parameter KAPPA is not correct")
    elif section_name == "TOTAL_VARIATION":
        for key, value in parser.items(section_name):
            if key.upper() == 'WEIGHT':
                if is_float(value):
                    totalvariation_paramenters[key.upper()] = float(value)
                else:
                    sys.exit("[tis.conf ERROR]: total_variation parameter WEIGHT is not correct")
            elif key.upper() == 'MAX_ITER':
                if is_float(value):
                    if float(value).is_integer():
                        totalvariation_paramenters[key.upper()] = int(value)
                    else:
                        sys.exit("[tis.conf ERROR]: anisodiff parameter MAX_ITER is not correct")
                else:
                    sys.exit("[tis.conf ERROR]: anisodiff parameter MAX_ITER is not correct")
            elif key.upper() == 'EPS':
                if is_float(value):
                    totalvariation_paramenters[key.upper()] = float(value)
                else:
                    sys.exit("[tis.conf ERROR]: total_variation parameter EPS is not correct")
            elif key.upper() == 'ISOTROPIC':
                if value.upper() == 'TRUE':
                    totalvariation_paramenters[key.upper()] = True
                elif value.upper() == 'FALSE':
                    totalvariation_paramenters[key.upper()] = False
                else:
                    sys.exit("[tis.conf ERROR]: total_variation parameter ISOTROPIC is not boolean")
    elif section_name == "WAVELET":
        for key, value in parser.items(section_name):
            if key.upper() == 'SIGMA_W':
                if value.upper() == 'NONE':
                    wavelet_paramenters[key.upper()] = None
                elif is_float(value):
                    wavelet_paramenters[key.upper()] = float(value)
                else:
                    sys.exit("[tis.conf ERROR]: wavelet parameter SIGMA_W is not correct")
            if key.upper() == 'WAVELET':
                if value.lower() in pywt.wavelist():
                    wavelet_paramenters[key.upper()] = value.lower()
                else:
                    sys.exit("[tis.conf ERROR]: wavelet parameter WAVELET is not correct")





"""
for section_name in parser.sections():
    if section_name in ["INPUT PARAMETERS", 
                        "OUTPUT PARAMETERS", 
                        "DATAMANAGER PARAMETERS", 
                        "SEGMENTATION PARAMETERS"]:
        for key, value in parser.items(section_name):
            if value.upper() == "NONE":
                value = None
            elif value.upper() == "TRUE":
                value = True
            elif value.upper() == "FALSE":
                value = False
            elif is_float(value):
                value = float(value)
                if value.is_integer():
                    value = int(value)
            parameters[key.upper()] = value
"""


def get_parameters():
    parameters_dict = dict()
    parameters_dict.update(input_parameters)
    parameters_dict.update(output_parameters)
    parameters_dict.update(datamanager_paramenters)
    parameters_dict.update(segmentation_parameters)

    if segmentation_parameters['DENOISE_TYPE'] == 'gaussian':
        parameters_dict.update(gaussian_parameters)
    elif segmentation_parameters['DENOISE_TYPE'] == 'anisodiff':
        parameters_dict.update(anisodiff_paramenters)
    elif segmentation_parameters['DENOISE_TYPE'] == 'total_variation':
        parameters_dict.update(totalvariation_paramenters)
    elif segmentation_parameters['DENOISE_TYPE'] == 'wavelet':
        parameters_dict.update(wavelet_paramenters)

    return parameters_dict


parameters = get_parameters()
print(parameters)

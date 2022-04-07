import configparser
import os

# TODO: Aggiungere controllo input da tis.conf

# TODO: aggiungere i tipi di filtro (kmeans (default), ps entropy.... con i relativi parametri
#       aggiungere i tipi di denoising (gaussian (default), anisodiff... con i relativi parametri

# DEFAULT PARAMETERS
parameters = {# Input Parameters
              "DATA_FOLDER": "data",
              "IMG_FILE": "img.fits",
              "RMS_FILE": "rms.fits",
              # Output Parameters
              "OUT_FOLDER": "data",
              "OUT_FILE": "segm.fits",
              "OUT_INFO": "segm.csv",
              # DataManager Parameters
              "DIM_PATCH": 512,
              "DIM_OVERLAP": 50,
              # Segmentation Parameters
              "T": 2,
              "D": 3,
              "SIGMA": None,
              "CONVEX": False,
              "ELLIPSE": False,
             }

# READ PARAMETERS FROM .CONF FILE
parser = configparser.ConfigParser()
config_file_path = os.path.join("", "tis.conf")
parser.read(config_file_path)


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False


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

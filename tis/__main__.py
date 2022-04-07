from utils import open_fits, write_fits, write_info
from datamanager import DataManager
from config import parameters

import os
import sys


print("*********************************************************")
print("***********************     *     ***********************")
print("                        *  TIS  *                        ")
print("***********************   *   *   ***********************")
print("*********************************************************")

print()
print("PARAMETERS")
print("----------")
for key, value in parameters.items():
    print(key, '=', value)
print()

DATA_FOLDER = parameters['DATA_FOLDER']
IMG_FILE = parameters['IMG_FILE']
RMS_FILE = parameters['RMS_FILE']

OUT_FOLDER = parameters['OUT_FOLDER']
OUT_FILE = parameters['OUT_FILE']
OUT_INFO = parameters['OUT_INFO']

data_folder = os.path.join('.', DATA_FOLDER)
out_folder = os.path.join('.', OUT_FOLDER)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

if len(os.listdir(out_folder)) != 0:
    sys.exit("Error: output folder is not empty.")

img_path = os.path.join(data_folder, IMG_FILE)
rms_path = os.path.join(data_folder, RMS_FILE)

seg_path = os.path.join(out_folder, OUT_FILE)
inf_path = os.path.join(out_folder, OUT_INFO)

print("START")
print("-----")

print("Reading fits...", end='')
img = open_fits(img_path)
rms = open_fits(rms_path)
print('DONE')

patcher = DataManager(dim=parameters['DIM_PATCH'],
                      overlap=parameters['DIM_OVERLAP'])

print('Patching...', end='')
patcher.patch(img, rms)
print('DONE!')

print('Processing...', end='')
patcher.process()
print('DONE!')

print('Unpatching...', end='')
patcher.unpatch()
print('DONE!')

segm = patcher.get_seg()
info = patcher.get_info()

print('Export segmentation...', end='')
write_fits(segm, seg_path)
print('DONE!')

print('Export info...', end='')
write_info(info, inf_path)
print('DONE!')

print()
print("BYE BYE!")

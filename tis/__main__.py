from .utils import open_fits, write_fits, write_info
from .datamanager import DataManager
from .config import parameters

import os
import sys
import argparse
import shutil


# ARGPARSER
parser = argparse.ArgumentParser(description='Topological Image Segmentation')

parser.add_argument('--export_config', type=str, default=None, help='Folder where to create the configuration file')
parser.add_argument('--img', type=str, default=None, help='Input image')
parser.add_argument('--rms', type=str, default=None, help='Input rms map')
parser.add_argument('--output', type=str, default=None, help='Output folder')
args = parser.parse_args()

if args.export_config is not None:
    src = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default_tis.py')
    dst = os.path.join(args.export_config, 'tis.conf')
    shutil.copyfile(src, dst)
    print('Message: tis.conf is created!')
    sys.exit()


print("*********************************************************")
print("***********************     *     ***********************")
print("                        *  TIS  *                        ")
print("***********************   *   *   ***********************")
print("*********************************************************")

DATA_FOLDER = parameters['DATA_FOLDER']
IMG_FILE = parameters['IMG_FILE']
RMS_FILE = parameters['RMS_FILE']

if args.output is None:
    OUT_FOLDER = parameters['OUT_FOLDER']
else:
    OUT_FOLDER = args.output
OUT_FILE = parameters['OUT_FILE']
OUT_INFO = parameters['OUT_INFO']

data_folder = DATA_FOLDER
out_folder = OUT_FOLDER

i = 0
while True:
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        break
    else:
        if i == 0:
            out_folder = out_folder + '_0'
        else:
            out_folder = out_folder[:-1] + str(i)
        i += 1

if args.img is None:
    img_path = os.path.join(data_folder, IMG_FILE)
else:
    img_path = args.img
if args.rms is None:
    rms_path = os.path.join(data_folder, RMS_FILE)
else:
    rms_path = args.rms


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

print("-----")
print("BYE BYE!")

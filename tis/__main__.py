from utils import open_fits, write_fits
from datamanager import DataManager

import os


data_folder = os.path.join('data')

grid_folder = os.path.join(data_folder, 'grid')
gsh160_folder = os.path.join(data_folder, 'GS_H160.crop')
mcvis_folder = os.path.join(data_folder, 'MC_vis')
sc8vis_folder = os.path.join(data_folder, 'SC8_vis.crop')

x_folder = grid_folder

img_path = os.path.join(x_folder, 'img.fits')
rms_path = os.path.join(x_folder, 'rms.fits')
segm_path = os.path.join(x_folder, 'segm.fits')


print("Reading fits...", end='')
img = open_fits(img_path)
rms = open_fits(rms_path)
try:
    true = open_fits(segm_path)
except:
    true = open_fits(os.path.join(mcvis_folder, 'true.fits'))
print('DONE')
print()

patcher = DataManager(dim=512, overlap=50)

print('Patching...', end='')
patcher.patch(img,rms)
print('DONE!')
print()
print('Processing...', end='')
patcher.process()
print('DONE!')
print()
print('Unpatching...', end='')
patcher.unpatch()
print('DONE!')
print()
        

print('END')
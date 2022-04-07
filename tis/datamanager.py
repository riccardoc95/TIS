import numpy as np
import pandas as pd

from preprocess import preprocess as pp
from segmentation import segmentation as sg
from config import parameters

import asyncio
import nest_asyncio

nest_asyncio.apply()


async def image_process(patch):
    patch.update(pp(patch.get_img(),
                    patch.get_rms(),
                    sigma=parameters['SIGMA'],
                    t=parameters['T'],
                    d=parameters['D']))
    seg, info = sg(patch.get_data(),
                   img=patch.get_img(),
                   rms=patch.get_rms(),
                   convex=parameters['CONVEX'],
                   ellipse=parameters['ELLIPSE'],
                   start_id=np.square(patch.x_end - patch.x_start) * patch.idx)
    patch.update_seg(seg)
    patch.update_info(pd.DataFrame(info))
    

class Patch:
    def __init__(self,
                 idx,
                 x_start, 
                 y_start, 
                 x_end, 
                 y_end, 
                 img, 
                 rms=None):
        # index
        self.idx = idx
        
        #coordinates
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end

        # data
        self.img = img
        self.rms = rms
        self.data = None
        self.seg = None
        self.info = None
        

    def update(self, data):
        self.data = data

    def update_seg(self, seg):
        self.seg = seg
    
    def update_info(self, info):
        self.info = info

    def get_coordinates(self):
        return self.x_start, self.x_end, self.y_start, self.y_end

    def get_img(self):
        return self.img
    
    def get_rms(self):
        return self.rms
    
    def get_data(self):
        return self.data

    def get_seg(self):
        return self.seg

    def get_info(self):
        return self.info


class DataManager:
    def __init__(self, dim, overlap):
        # settings
        self.dim = dim
        self.overlap = overlap
        self.step = dim - overlap
        
        # data
        self.data = []
        self.segment = None
        self.info = None
        
    def get_data(self):
        return self.data
    
    def get_seg(self):
        return self.segment
    
    def get_info(self):
        return self.info

    def patch(self, img, rms):
        n, m = img.shape

        # padding
        self.x_dim = n + 2 * self.overlap
        self.y_dim = m + 2 * self.overlap
        self.x_pad = [self.overlap,
                      self.overlap + (self.dim - (self.x_dim % self.step))]
        self.y_pad = [self.overlap,
                      self.overlap + (self.dim - (self.y_dim % self.step))]
        img_pad = np.pad(img, [self.x_pad, self.y_pad],
                         mode='constant', constant_values=(np.nan,))
        rms_pad = np.pad(rms, [self.x_pad, self.y_pad],
                         mode='constant', constant_values=(np.nan,))
        self.x_dim += (self.dim - (self.x_dim % self.step))
        self.y_dim += (self.dim - (self.y_dim % self.step))
        
        count = 0
        for x in range(0, self.x_dim - self.step, self.step):
            for y in range(0, self.y_dim-self.step, self.step):
                new_patch = Patch(idx = count,
                                  x_start=x,
                                  y_start=y,
                                  x_end=x+self.dim,
                                  y_end=y+self.dim,
                                  img=img_pad[x:x+self.dim, y:y+self.dim],
                                  rms=rms_pad[x:x+self.dim, y:y+self.dim])
                self.data.append(new_patch)
                count += 1
            
    def process(self):
        loop = asyncio.get_event_loop()
        tasks = [image_process(x) for x in self.data]
        loop.run_until_complete(asyncio.wait(tasks))

    
    def _fix_info_position(self, patch):
        info_patch = patch.get_info()
        x_start, x_end, y_start, y_end = patch.get_coordinates()
        info_patch['x'] = info_patch['x'] + x_start
        info_patch['y'] = info_patch['y'] + y_start
        info_patch['x_min'] = info_patch['x_min'] + x_start
        info_patch['y_min'] = info_patch['y_min'] + y_start
        info_patch['x_max'] = info_patch['x_max'] + x_start
        info_patch['y_max'] = info_patch['y_max'] + y_start
        return info_patch
        
    def unpatch(self):
        self.segment = np.zeros((self.x_dim, self.y_dim))
        self.info = pd.DataFrame([])
        
        n_elem = 1
        for patch in self.data:
            x_s, x_e, y_s, y_e = patch.get_coordinates()
            
            seg_patch = patch.get_seg()
            res_patch = self.segment[x_s:x_e, y_s:y_e]
            info_patch = self._fix_info_position(patch)
            img = patch.get_img()
            rms = patch.get_rms()
            
            for idx in info_patch['id']:
                # intersezioni
                vals, counts = np.unique(res_patch[seg_patch == idx], return_counts=True)
                vals = vals[np.argsort(-counts)]
                vals = np.setdiff1d(vals, np.array([0]))
                
                if len(vals) == 0:
                    self.segment[x_s:x_e, y_s:y_e][seg_patch == idx] = n_elem
                    info = info_patch[info_patch['id'] == idx]
                    info['id'] = n_elem
                    self.info = pd.concat([self.info, info])
                    n_elem +=1
                else:
                    info = info_patch[info_patch['id'] == idx]
                    info['id'] =  vals[0]
                    for v in vals:
                        w = np.where((res_patch == v) & (seg_patch == idx))
                        seg_patch[w] = 0
                        info['area'] = info['area'] - len(w[0])
                        info['flusso'] = info['flusso'] - img[w].sum()
                        info['errore'] = info['errore'] - (rms[w]**2).sum()
                    
                    self.segment[x_s:x_e, y_s:y_e][seg_patch == idx] = vals[0]
                    self.info = pd.concat([self.info, info])
                    
                
        # segmentation
        self.segment = self.segment[self.x_pad[0]:-self.x_pad[1], 
                                    self.y_pad[0]:-self.y_pad[1]]
        # info    
        self.info = self.info.groupby(['id']).agg({'x':lambda x: x.mean() - self.x_pad[0],
                                                   'y':lambda x: x.mean() - self.y_pad[0],
                                                   'x_min':lambda x: x.min() - self.x_pad[0],
                                                   'y_min':lambda x: x.min() - self.y_pad[0],
                                                   'x_max':lambda x: x.max() - self.x_pad[0],
                                                   'y_max':lambda x: x.max() - self.y_pad[0], 
                                                   'area':'sum',
                                                   'flusso':'sum',
                                                   'errore':'sum',}) 
        self.info = self.info.reset_index()
        
        self.info[['id', 'x', 'y', 'x_min', 'y_min', 'x_max', 'y_max']] = \
        self.info[['id', 'x', 'y', 'x_min', 'y_min', 'x_max', 'y_max']].astype(int)
        self.info[['area', 'flusso', 'errore']] = \
        self.info[['area', 'flusso', 'errore']].astype(float)
       
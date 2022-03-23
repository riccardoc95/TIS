import numpy as np
import pandas as pd
import asyncio

from preprocess import preprocess as pp
from segmentation import segmentation as sg

from tqdm import tqdm

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def image_process(patch):
    patch.update(pp(patch.get_img(), patch.get_rms()))
    seg, info = sg(patch.get_data(), patch.get_img(), patch.get_rms(), 
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
        for x in self.data:
            image_process(x)
    
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
        
        count = 1
        mapping = {}
        mapping[0] = 0
        
        for patch in tqdm(self.data):
            x_s, x_e, y_s, y_e = patch.get_coordinates()
             
            if self.segment[x_s:x_e, y_s:y_e].sum() == 0:
                self.segment[x_s:x_e, y_s:y_e] = patch.get_seg()
                
                info_patch = self._fix_info_position(patch)
                self.info = pd.concat([self.info, info_patch])
                
            else:
                seg_patch = patch.get_seg()
                res_patch = self.segment[x_s:x_e, y_s:y_e]
                
                idxs = np.unique(res_patch[seg_patch > 0])
                idxs = np.setdiff1d(idxs,np.array([0]))
                
                info_patch = self._fix_info_position(patch)
                info_patch.set_index('id', inplace=True)
                img = patch.get_img()
                rms = patch.get_rms()

                for idx in idxs:
                    unique = np.unique(seg_patch[res_patch == idx])
                    val = unique[0] if unique[0] != 0 else unique[1]
                    
                    # Info
                    w = np.where((seg_patch == val) & (res_patch == idx))
                    info_patch.loc[int(val),'area'] = info_patch.loc[int(val),'area'] - len(w[0])
                    info_patch.loc[int(val),'flusso'] = info_patch.loc[int(val),'flusso'] - img[w].sum()
                    info_patch.loc[int(val),'errore'] = info_patch.loc[int(val),'errore'] - (rms[w]**2).sum()

                    if idx not in mapping.keys():
                        mapping[int(idx)] = int(count)
                        mapping[int(val)] = int(count)
                        count += 1
                    else:
                        mapping[int(val)] = mapping[int(idx)]
                
                w = np.where(self.segment[x_s:x_e, y_s:y_e] == 0)
                self.segment[x_s:x_e, y_s:y_e][w] = patch.get_seg()[w]

                info_patch.reset_index(inplace=True)
                self.info = pd.concat([self.info, info_patch])
        
        idxs = np.unique(self.segment)
        idxs = np.setdiff1d(idxs,np.array([0]))
        for idx in tqdm(idxs):
            if idx not in mapping.keys():
                mapping[int(idx)] = int(count)
                count += 1
        

        # segmentation
        k = np.array(list(mapping.keys()))
        v = np.array(list(mapping.values()))
        mapping_ar = np.zeros(k.max()+1,dtype=v.dtype) 
        mapping_ar[k] = v
        self.segment = mapping_ar[self.segment.astype(int)]
        self.segment = self.segment[self.x_pad[0]:-self.x_pad[1], 
                                    self.y_pad[0]:-self.y_pad[1]]
        
        # info
        self.info = self.info.replace({"id": mapping})
        
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

       
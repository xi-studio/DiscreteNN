import glob
import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import os
import random
import librosa
import pyworld as pw

def coded_sp_padding(coded_sp, multiple = 4):

    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values = 0)

    return coded_sp_padded

def default_loader(path):
    x, fs = librosa.load(path, sr=16000, dtype=np.float64)
    f0, sp, ap = pw.wav2world(x, fs)
    csp = pw.code_spectral_envelope(sp, fs, 24)
    csp = coded_sp_padding(csp.T)
    csp = csp.astype(np.float32)
    
    return csp, f0


class Audio(data.Dataset):
    def __init__(self, name='train'):
        super(Audio, self).__init__()
        self.image_list = glob.glob('/data/tree/qinghua/%s/*.wav' % name)

    def __getitem__(self, index):
        path = self.image_list[index]
        csp, f0 = default_loader(path) 
        
        ID = 0
        if '8_' in path:
            ID = 1


        return csp, f0, ID

    def __len__(self):

        return len(self.image_list)



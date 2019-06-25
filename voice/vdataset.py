import glob
import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import os
import random
import librosa

def default_loader(path):
    x, sr = librosa.load(path, sr=16000)
    k = x.shape[0] // 800
    img = x[:800 * k]
    img = img.reshape(k, 800)
    
    
    return img


class Audio(data.Dataset):
    def __init__(self, name='train'):
        super(Audio, self).__init__()
        self.image_list = glob.glob('/data/tree/qinghua/%s/*.wav' % name)

    def __getitem__(self, index):
        path = self.image_list[index]
        audio = default_loader(path) 
        
        ID = 0
        if '8_' in path:
            ID = 1


        return audio, ID

    def __len__(self):

        return len(self.image_list)



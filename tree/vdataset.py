import glob
import h5py
import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import os
import random
import librosa
import pyworld as pw

img_data, target = torch.load('./data/train.pt')
img_data = img_data.type(torch.FloatTensor)
img_data = img_data / 256.0

def loadfile(path):
    f = h5py.File(path,'r')
    x = f['randx'][:]
    f.close()
    return x

def default_loader(idx):
    img = img_data[idx]
    path = '/data/tree/mnist/%d.h5' % idx
    x = loadfile(path)

    return x, img 


class Audio(data.Dataset):
    def __init__(self):
        super(Audio, self).__init__()

    def __getitem__(self, index):
        x, img = default_loader(index) 

        return x, img

    def __len__(self):

        return 60000 


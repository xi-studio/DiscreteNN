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

img_data, target = torch.load('./data/train.pt')

randx = torch.randn(60000, 10, 100, 100)

def default_loader(idx):
    img = img_data[idx]
    x = randx[idx]

    return x, img 


class Audio(data.Dataset):
    def __init__(self):
        super(Audio, self).__init__()

    def __getitem__(self, index):
        x, img = default_loader(index) 

        return x, img

    def __len__(self):

        return 60000 


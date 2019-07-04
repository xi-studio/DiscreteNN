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
    idx = np.random.randint(0, x.shape[0] - 16000)
    img = x[idx: idx + 16000]

    img = mu_law_encoding(img, 256)
    img = quantize_data(img, 256)

    img = img.reshape(40, 400)

    
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


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized

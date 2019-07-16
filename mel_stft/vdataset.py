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

def default_loader(path):
    x, fs = librosa.load(path, sr=16000)
    idx = 12000
    if 'train' in path:
        idx = np.random.randint(0, x.shape[0] - 16000 * 2 - 128 * 3)
    y = x[idx: idx + 16000 * 2 - 128 * 3]

    mel = librosa.feature.melspectrogram(y, sr=fs, n_fft=1024, hop_length=128) 

    mel = np.log(mel)
    mel = mel.astype(np.float32)
    ID = np.zeros((2, mel.shape[1]), dtype=np.float32)

    return mel, ID 


class Audio(data.Dataset):
    def __init__(self, name='train'):
        super(Audio, self).__init__()
        self.image_list = glob.glob('/data/tree/qinghua/%s/*.wav' % name)

    def __getitem__(self, index):
        path = self.image_list[index]
        mel, ID = default_loader(path) 
        
        if '4_' in path:
            ID[0, :] = 1
        else:
            ID[1, :] = 1

        return mel, ID 

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

def logf0_statistics(f0s):

    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted

def decode_csp(coded_sp, fs):
    fftlen = pw.get_cheaptrick_fft_size(fs)
    decoded_sp = pw.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp

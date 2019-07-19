from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from vmodel import *
from vdataset import *

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

statistics = np.load('data/statistics.npy')


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = DataLoader(Audio('train'),batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(Audio('test'),batch_size=args.batch_size, shuffle=True, num_workers=4)


model = VAE()
model.load_state_dict(torch.load('/data/tree/voice/mel_f_880.pt'))
model = model.to(device)
print('# parameters:', sum(param.numel() for param in model.parameters()) /1000000.0 * 4)


loss_function = nn.MSELoss()


def savefig(path, mel):
    plt.figure(figsize=(10, 5))
    plt.imshow(mel)
    plt.savefig(path, format="png")
    plt.close()

def load_audio(path):
    x, fs = librosa.load(path, sr=16000) 
    mel = librosa.feature.melspectrogram(x, sr=fs, n_fft=1024, hop_length=128) 
    mel = np.log(mel)
    mel = mel.astype(np.float32)

    return mel, x 

def rec_audio(mel):
    M = np.exp(mel)
    S = librosa.feature.inverse.mel_to_stft(M, sr=16000, n_fft=1024)
    print(S.shape)
    y = librosa.griffinlim(S, hop_length=128)

    return y

def gen():
    model.eval()
    with torch.no_grad():
        path = '/data/tree/qinghua/train/A8_240.wav'
        mel, y = load_audio(path)
        L = mel.shape[1]

        c_4 = np.zeros((1, 2), dtype=np.float32)
        c_8 = np.zeros((1, 2), dtype=np.float32)

        c_4[:,0] = 1
        c_8[:,1] = 1

        data = torch.from_numpy(mel)
        c_4  = torch.from_numpy(c_4)
        c_8  = torch.from_numpy(c_8)

        x    = data.to(device)
        c_4  = c_4.to(device)
        c_8  = c_8.to(device)

        x = x.unsqueeze(0)

        t = torch.arange(100)
        t = t.type(torch.FloatTensor)
        t = t.to(device)

        rx_4 = model(x, c_4, t)
        rx_8 = model(x, c_8, t)
        
        print(x.shape, rx_4.shape, rx_8.shape)

        img = torch.cat((x, rx_4, rx_8), dim=1)

        savefig('images/generate_re_mix.png', img.cpu().numpy()[0])

        y1 = rec_audio(mel)
        y2 = rec_audio(rx_4.cpu().numpy()[0])
        y3 = rec_audio(rx_8.cpu().numpy()[0])
        
        fs = 16000
        librosa.output.write_wav('images/mix_a_8_A_240_org.wav', y, sr=fs)
        librosa.output.write_wav('images/mix_a_8_A_240_grm_4.wav', y1, sr=fs)
        librosa.output.write_wav('images/mix_a_8_A_240_rx_4.wav', y2, sr=fs)
        librosa.output.write_wav('images/mix_a_8_A_240_rx_8.wav', y3, sr=fs)

if __name__ == "__main__":
    gen()

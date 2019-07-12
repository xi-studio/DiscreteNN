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
model.load_state_dict(torch.load('/data/tree/voice/csp_wav_300.pt'))
model = model.to(device)
print('# parameters:', sum(param.numel() for param in model.parameters()) /1000000.0 * 4)


loss_function = nn.MSELoss()


def savefig(path, mel, mel1, mel2):
    plt.figure(figsize=(10, 5))
    plt.plot(mel)
    plt.plot(mel1)
    plt.plot(mel2)
    plt.savefig(path, format="png")
    plt.close()

def load_audio(path):
    x, fs = librosa.load(path, sr=16000, dtype=np.float64) 

    f0, sp, ap= pw.wav2world(x, fs)
    csp = pw.code_spectral_envelope(sp, fs, 24)

    csp = csp.T
    csp = csp.astype(np.float32)
    L = csp.shape[1]

    csp = csp.reshape(1, 24, L)

    return csp, f0, L, sp, ap, fs 

def gen():
    model.eval()
    with torch.no_grad():
        path = '/data/tree/qinghua/train/A4_240.wav'
        csp, f0, L, sp, ap, fs = load_audio(path)

        c_4 = np.zeros((1, 2, L), dtype=np.float32)
        c_4[:, 0, :] = 1
        c_8 = np.zeros((1, 2, L), dtype=np.float32)
        c_8[:, 1, :] = 1

        f0_8_org = pitch_conversion(f0, statistics[0][0], statistics[0][1], statistics[1][0], statistics[1][1])

        data = torch.from_numpy(csp)
        c_4  = torch.from_numpy(c_4)
        c_8  = torch.from_numpy(c_8)

        x    = data.to(device)
        c_4  = c_4.to(device)
        c_8  = c_8.to(device)

        t = torch.arange(100)
        t = t.type(torch.FloatTensor)
        t = t.to(device)

        rx_4 = model(x, c_4, t)
        rx_8 = model(x, c_8, t)
        
        print(rx_4.shape, rx_8.shape)

        img = torch.cat((x, rx_4, rx_8), dim=1)
        img = img.unsqueeze(1)
        save_image(img.cpu(),
                         'images/generate_mix_4_A_240.png', nrow=1)

        rx_4_csp = rx_4.cpu().numpy()[0]
        rx_4_csp = rx_4_csp.T 
        rx_4_csp = np.ascontiguousarray(rx_4_csp.astype(np.float64))

        rx_8_csp = rx_8.cpu().numpy()[0]
        rx_8_csp = rx_8_csp.T 
        rx_8_csp = np.ascontiguousarray(rx_8_csp.astype(np.float64))
        dsp_4 = decode_csp(rx_4_csp, fs)
        dsp_8 = decode_csp(rx_8_csp, fs)
        y = pw.synthesize(f0, sp, ap, fs)
        y4 = pw.synthesize(f0, dsp_4, ap, fs)
        y4_1 = pw.synthesize(f0, dsp_8, ap, fs)
        y8 = pw.synthesize(f0_8_org, dsp_8, ap, fs)
        y8_1 = pw.synthesize(f0_8_org, dsp_4, ap, fs)
                
        librosa.output.write_wav('images/mix_4_A_240_org.wav', y, sr=fs)
        librosa.output.write_wav('images/mix_4_A_240_rx_4.wav', y4, sr=fs)
        librosa.output.write_wav('images/mix_4_A_240_rx_4_1.wav', y4_1, sr=fs)
        librosa.output.write_wav('images/mix_4_A_240_rx_8.wav', y8, sr=fs)
        librosa.output.write_wav('images/mix_4_A_240_rx_8_1.wav', y8_1, sr=fs)

if __name__ == "__main__":
    gen()

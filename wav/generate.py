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


model = VAE()
model.load_state_dict(torch.load('/data/tree/voice/phase_200.pt'))
model = model.to(device)

print('# parameters:', sum(param.numel() for param in model.parameters()) /1000000.0 * 4)


loss_function = nn.CrossEntropyLoss()


def savefig(path, mel, mel1, mel2):
    plt.figure(figsize=(15, 5))
    plt.plot(mel)
    plt.plot(mel1)
    plt.plot(mel2)
    plt.savefig(path, format="png")
    plt.close()


def load_audio(path):
    x, sr = librosa.load(path, sr=16000) 
    k = x.shape[0] // 400
    y = x[: 400 * k]
    y = y.reshape(1, k, 400)
    data = mu_law_encoding(y, 256)  
    data = quantize_data(data, 256)

    ID = np.ones((1, k, 1), dtype=np.float32)

    if '8_' in path:
        ID = ID * 2
     
    return data, ID

   
def gen():
    model.eval()
    with torch.no_grad():
        path = '/data/tree/qinghua/train/A4_240.wav'
        data, c = load_audio(path)
        data = torch.from_numpy(data)
        c = torch.from_numpy(c)

        data = data.view(-1)
        
        x_hot = torch.zeros(data.shape[0], 256)
        x_hot.scatter_(1, data.unsqueeze(1), 1)

        c = c.to(device)
        x_hot = x_hot.to(device)

        x_hot = x_hot.view(-1, 400, 256)
        x_hot = x_hot.transpose(1, 2).contiguous()

        t = torch.arange(100)
        t = t.type(torch.FloatTensor)
        t = t.to(device)


        rx = model(x_hot, c, t)
        rx = rx.transpose(1, 2).contiguous()
        rx = rx.view(-1, 256)

        c1 = c * 2 

        smp = model(x_hot, c1, t)
        smp = smp.transpose(1, 2).contiguous()
        smp = smp.view(-1, 256)

        rx = rx.argmax(dim=1)
        smp = smp.argmax(dim=1)
        print(data.shape, rx.shape)


        rx = rx.cpu().numpy()
        smp = smp.cpu().numpy()
        data = data.numpy()


        wav_rx = (rx /256) * 2.0 -1
        wav_rx = mu_law_expansion(wav_rx, 256)
        wav_smp = (smp /256) * 2.0 -1
        wav_smp = mu_law_expansion(wav_smp, 256)
        wav_data = (data /256) * 2.0 -1
        wav_data = mu_law_expansion(wav_data, 256)

        librosa.output.write_wav('images/4_A_240_org.wav', wav_data, sr=16000)
        librosa.output.write_wav('images/4_A_240_rec.wav', wav_rx, sr=16000)
        librosa.output.write_wav('images/4_A_240_B.wav', wav_smp, sr=16000)


       
        savefig('images/gen_w_4_A_240.png', wav_data, wav_rx + 2, wav_smp + 4)
#                
#    test_loss /= len(test_loader.dataset)
#    print('====> Test set loss: {:.4f} '.format(test_loss))
#
if __name__ == "__main__":
     gen()

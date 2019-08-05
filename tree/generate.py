from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)

norm = torch.nn.functional.normalize

class GLU(nn.Module):
    def __init__(self, c1, c2):
        super(GLU, self).__init__()
        self.s = nn.Linear(c1, c2)
        self.g = nn.Linear(c1, c2)

    def forward(self, x):
        s = torch.sigmoid(self.s(x))
        g = torch.relu(self.g(x))
        output = s * g

        return output 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 50)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        phase = torch.sigmoid(self.fc2(x))

        return phase

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = GLU(100, 400)
        self.fc2 = nn.Linear(400, 784)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.sigmoid(self.fc2(x))

        return x 


class Key(nn.Module):
    def __init__(self):
        super(Key, self).__init__()

        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 50)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        w = torch.sigmoid(self.fc2(x))

        return w 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.e = Encoder()
        self.d = Decoder()
        self.amplitude = Key()

    def forward(self, x, c, t):
        x = x.view(-1, 784)
        N = x.shape[0]

        w = self.amplitude(c)
        phase = self.e(x)

        w = w.view(N, 50, 1)
        phase = phase.view(N, 50, 1)
       
        w = w.repeat(1, 1, 100)
        phase = phase.repeat(1, 1, 100)

        x = torch.sin(2 * np.pi * w * t  + np.pi * phase )
        x = x.sum(dim=1)
        x = x.view(N, 100)         
        noise = torch.randn_like(x)
        x = noise + x
        x = self.d(x)

        return x, w, phase

model = VAE().to(device)
model.load_state_dict(torch.load('checkpoints/mnist/fft_400.pt'))


def test():
    model.eval()
    with torch.no_grad():
            t = torch.arange(100)
            t = t.type(torch.FloatTensor)
            t = t.to(device)

            c = torch.zeros(64, 10).to(device)
            c[:, 4] =1
            data = torch.rand(64, 1, 28, 28).to(device)
            rx, w, phase= model(data, c, t)
            img = rx.view(64, 1, 28, 28)
            save_image(img.cpu(),
                         'images/sample_4.png', nrow=8)
              
        
#            for i in range(100):
#                rx, w, phase= model(data, c, t)
#                img = rx.view(1, 1, 28, 28)
#                save_image(img.cpu(),
#                         'images/sample_t_%d.png' % i, nrow=1)
#                data = rx
#
if __name__ == "__main__":
    test()

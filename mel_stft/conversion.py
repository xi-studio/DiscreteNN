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

train_loader = DataLoader(Audio('train'),batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(Audio('test'),batch_size=args.batch_size, shuffle=False, num_workers=4)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

print('# parameters:', sum(param.numel() for param in model.parameters()) /1000000.0 * 4)


loss_function = nn.MSELoss()


def savefig(path, mel):
    plt.figure(figsize=(4, 8))
    plt.imshow(mel)
    plt.savefig(path, format="png")
    plt.close()

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, c) in enumerate(train_loader):
        x = data.to(device)
        c = c.to(device)

        t = torch.arange(100)
        t = t.type(torch.FloatTensor)
        t = t.to(device)

        optimizer.zero_grad()
       
        rx = model(x, c, t)
        loss = loss_function(rx, x)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data), 
                ))

    if epoch % 20 == 0:
        torch.save(model.state_dict(),"/data/tree/voice/mel_f_%03d.pt" % epoch)
    

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss /len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, c) in enumerate(test_loader):
            x = data.to(device)
            c = c.to(device)

            c1 = torch.zeros_like(c)
            c2 = torch.zeros_like(c)
            c1[:,0] = 1
            c2[:,1] = 1

            t = torch.arange(100)
            t = t.type(torch.FloatTensor)
            t = t.to(device)

            rx = model(x, c, t)
            loss = loss_function(rx, x)
            test_loss += loss.item()

            if i == 0:
                rx1 = model(x, c1, t)
                rx2 = model(x, c2, t)
                img = torch.cat((x, rx, rx1, rx2), dim=1)
                savefig('images/mel_f_%03d.png' % epoch, img.cpu().numpy()[0])
                
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} '.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

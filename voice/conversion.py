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
test_loader = DataLoader(Audio('test'),batch_size=args.batch_size, shuffle=True, num_workers=4)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('# parameters:', sum(param.numel() for param in model.parameters()) /1000000.0 * 4)

def savefig(path, mel, mel1):
    plt.figure(figsize=(10, 5))
    plt.plot(mel)
    plt.plot(mel1)
    plt.savefig(path, format="png")
    plt.close()


def loss_function(recon_x, x):
    BCE = F.mse_loss(recon_x, x)

    return BCE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, f0, c) in enumerate(train_loader):
#        N = data.shape[0]
#        c = c.repeat(1, N)
#        c = c.transpose(0, 1)
#
#        c_onehot = torch.zeros(data.shape[0], 10)
#        c_onehot.scatter_(1, c, 1)
#        c = c_onehot.to(device)
#
#        t = torch.arange(100)
#        t = t.type(torch.FloatTensor)
#        t = t.to(device)
#
        data = data.to(device)
        f0 = f0.to(device)
        optimizer.zero_grad()
       
        rx = model(data, f0)
        loss = loss_function(rx, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data), 
                ))

#    if epoch % 20 == 0:
#        torch.save(model.state_dict(),"checkpoints/voice/fft_%03d.pt" % epoch)
#    
#
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss /len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    test_similarity = 0
    with torch.no_grad():
        for i, (data, c) in enumerate(test_loader):
            data = data.view(-1, 800)
            N = data.shape[0]
            c = c.repeat(1, N)
            c = c.transpose(0, 1)

            c_onehot = torch.zeros(N, 10)
            c_onehot.scatter_(1, c, 1)
            c = c_onehot.to(device)
            t = torch.arange(100)
            t = t.type(torch.FloatTensor)
            t = t.to(device)

            data = data.to(device)
        
            rx, w, phase= model(data, c, t)
            loss = loss_function(rx, data)
            test_loss += loss.item()

            img = rx.view(-1)
            img1 = data.view(-1)
            
            img = img.cpu().numpy()
            img1 = img1.cpu().numpy() + 2
            if i==0: 
                savefig('images/wav_%03d.png' % epoch, img, img1)
            
            
#            if i == 0:
#                c1 = torch.zeros_like(c)
#                c1[:,6] = 1
#
#                w1 = model.amplitude(c1)
#                w1 = w1.view(data.shape[0], 50, 1)
#                w1 = w1.repeat(1, 1, 100)
#                
#                x =  torch.sin(2 * np.pi * w1 * t + np.pi * phase)
#                img = x
#                x = x.sum(dim=1)
#                x = model.d(x)
#               
#                n = min(data.size(0), 16)
#                img1 = torch.cat((data[:64].view(-1,784), rx[:64], x[:64]),dim=0)
#                img1 = img1.view(64 * 3, 1, 28, 28)
#                img = img[:4]
#                img = img.view(4, 1, 50, 100)
#                save_image(img.cpu(),
#                         'images/fft_08_' + str(epoch) + '.png', nrow=1)
#                save_image(img1.cpu(),
#                         'images/z_08_' + str(epoch) + '.png', nrow=64)
#
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} '.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
#        test(epoch)

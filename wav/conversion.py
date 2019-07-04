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


loss_function = nn.CrossEntropyLoss()


def savefig(path, mel, mel1):
    plt.figure(figsize=(10, 5))
    plt.plot(mel)
    plt.plot(mel1)
    plt.savefig(path, format="png")
    plt.close()

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, c) in enumerate(train_loader):
        x = data.view(-1)

        x_hot = torch.zeros(x.shape[0], 256)
        x_hot.scatter_(1, x.unsqueeze(1), 1)

        x = x.to(device)
        x_hot = x_hot.to(device)
       
        x_hot = x_hot.view(-1, 400, 256)
        x_hot = x_hot.transpose(1, 2)

        t = torch.arange(100)
        t = t.type(torch.FloatTensor)
        t = t.to(device)

        optimizer.zero_grad()
       
        rx = model(x_hot, c, t)
        rx = rx.transpose(1, 2)
        rx = rx.contiguous()
        loss = loss_function(rx.view(-1, 256), x)
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
    with torch.no_grad():
        for i, (data, c) in enumerate(test_loader):
            x = data.view(-1)

            x_hot = torch.zeros(x.shape[0], 256)
            x_hot.scatter_(1, x.unsqueeze(1), 1)

            x = x.to(device)
            x_hot = x_hot.to(device)
       
            x_hot = x_hot.view(-1, 400, 256)
            x_hot = x_hot.transpose(1, 2)

            t = torch.arange(100)
            t = t.type(torch.FloatTensor)
            t = t.to(device)
       
            rx = model(x_hot, c, t)
            rx = rx.transpose(1, 2)
            rx = rx.contiguous()
            rx = rx.view(-1, 256)
            loss = loss_function(rx, x)

            test_loss += loss.item()
            rx = rx.argmax(dim=1)

            
            if i == 0:
                x = x.view(args.batch_size, 16000)
                rx = rx.view(args.batch_size, 16000)
                savefig('images/test_%03d.png' % epoch, x[0].cpu().numpy(), rx[0].cpu().numpy() + 260)
                
#                img = torch.cat((data, rx),dim=1)
#                img = img.unsqueeze(1)
#                print(img.shape)
#                save_image(img.cpu(),
#                         'images/csp_' + str(epoch) + '.png', nrow=1)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} '.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

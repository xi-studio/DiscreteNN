from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from vdataset import *

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

train_loader = DataLoader(Audio('train'),batch_size=args.batch_size, shuffle=True, num_workers=4)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(400, 30)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        phase = torch.sigmoid(self.fc3(x))

        return phase

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.e = Encoder()

    def forward(self, x, c, t):
        x = x.view(-1, 784)
        N = x.shape[0]

        return x

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    return BCE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, img) in enumerate(train_loader):
         print(data.shape,img.shape)
         break


#        data = data.to(device)
#        optimizer.zero_grad()
#        
#        rx, w, phase= model(data, c, t)
#        loss = loss_function(rx, data)
#        loss.backward()
#        train_loss += loss.item()
#        optimizer.step()
#
#        if batch_idx % args.log_interval == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader),
#                loss.item() / len(data), 
#                ))
#
#    if epoch % 20 == 0:
#        torch.save(model.state_dict(),"checkpoints/mnist/rtest_%03d.pt" % epoch)
#    
#
#    print('====> Epoch: {} Average loss: {:.4f}'.format(
#          epoch, train_loss /len(train_loader.dataset)))
#

def test(epoch):
    model.eval()
    with torch.no_grad():
        save_image(img1.cpu(),'images/rtest2_' + str(epoch) + '.png', nrow=32)


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

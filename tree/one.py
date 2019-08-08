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

        

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
         
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 1)

        self.fc3 = nn.Linear(11, 1000)
        self.fc4 = nn.Linear(1000, 784)
        self.t = nn.Parameter(torch.linspace(0.01, 10, 1000),requires_grad=False)

    def forward(self, x, c):
        x = x.view(-1, 784)
        x = torch.sigmoid(self.fc1(x))
        x = x * self.t
        x = self.fc2(x)

        z = x
        x = torch.cat((x, c), dim=1)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x, z 

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    return BCE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, c) in enumerate(train_loader):

        c = c.unsqueeze(1)
        c_onehot = torch.zeros(data.shape[0], 10)
        c_onehot.scatter_(1, c, 1)
        c = c_onehot.to(device)

        data = data.to(device)
        optimizer.zero_grad()
        
        rx , z = model(data, c)
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

    if epoch % 20 == 0:
        torch.save(model.state_dict(),"checkpoints/mnist/one_%03d.pt" % epoch)
    

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss /len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, c) in enumerate(test_loader):
            c = c.unsqueeze(1)
            c_onehot = torch.zeros(data.shape[0], 10)
            c_onehot.scatter_(1, c, 1)
            c = c_onehot.to(device)

            data = data.to(device)
        
            rx, z = model(data, c)
            loss = loss_function(rx, data)
            test_loss += loss.item()
            if i == 0:
                c1 = torch.zeros_like(c)
                c1[:,6] = 1
                print(z[:10])

                #x = model(data, c1, t)
                
                #input_data = torch.rand_like(data)
                #x1 = model(input_data, c1, t)
               
                img = torch.cat((data[:32].view(-1,784), rx[:32].view(-1,784)),dim=0)
                img = img.view(32 * 2, 1, 28, 28)
                save_image(img.cpu(),
                         'images/one_' + str(epoch) + '.png', nrow=32)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} '.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

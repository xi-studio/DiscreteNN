from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def savefig(path, mel):
    plt.figure(figsize=(8, 4))
    plt.imshow(mel)
    plt.savefig(path, format="png")
    plt.close()

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

soft = nn.Softmax(dim=2)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 30)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        z = torch.sigmoid(self.fc2(x))

        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(1040, 1000)
        self.w1 = torch.nn.Parameter(data=torch.rand(50, 784),requires_grad=True) 
        self.z1 = torch.nn.Parameter(data=torch.rand(1, 1000),requires_grad=True) 

        self.conv = nn.Conv1d(20, 50, 1)
        self.conv1 = nn.Conv1d(50, 1, 1)

    def forward(self, x):
        N = x.shape[0]
        z = self.z1.repeat(N, 1)
        x = torch.cat((x, z), dim=1)

        x = self.fc1(x)
        x = x.view(-1, 20, 50)
        x = soft(x)
        x = x.matmul(self.w1)
        x = self.conv(x)
        x = self.conv1(x)
        x = torch.sigmoid(x)

        return x 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.e = Encoder()
        self.d = Decoder()

    def forward(self, x, c):
        x = x.view(-1, 784)
        x = self.e(x)
        x = torch.cat((x, c), dim=1)
        x = self.d(x)
        w = self.d.w1
        return x, w 

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

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
        
        rx, w = model(data, c)

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
        torch.save(model.state_dict(),"checkpoints/mnist/vq_%03d.pt" % epoch)
    

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss /len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    test_similarity = 0
    with torch.no_grad():
        for i, (data, c) in enumerate(test_loader):
            c = c.unsqueeze(1)
            c_onehot = torch.zeros(data.shape[0], 10)
            c_onehot.scatter_(1, c, 1)
            c = c_onehot.to(device)

            data = data.to(device)

            rx, w = model(data, c)
            rx = rx.view(-1, 784)
            loss = loss_function(rx, data)
            test_loss += loss.item()
            if i == 0:
                c1 = torch.zeros_like(c)
                c1[:,6] = 1

                sample, _ = model(data, c1)
                sample = sample.view(-1, 784)
                data = data.view(-1, 784)
               
                n = min(data.size(0), 16)
                img1 = torch.cat((data[:32], rx[:32], sample[:32]),dim=0)
                img1 = img1.view(32 * 3, 1, 28, 28)
                save_image(img1.cpu(),
                         'images/vq_' + str(epoch) + '.png', nrow=32)
                savefig('images/w_' + str(epoch) + '.png', w.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} '.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

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
        self.fc3 = nn.Linear(400, 30)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        phase = torch.sigmoid(self.fc3(x))

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
        self.fc2 = nn.Linear(50, 30)

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

        w = w.view(N, 30, 1)
        phase = phase.view(N, 30, 1)
       
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

        t = torch.arange(100)
        t = t.type(torch.FloatTensor)
        t = t.to(device)

        data = data.to(device)
        optimizer.zero_grad()
        
        rx, w, phase= model(data, c, t)
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
        torch.save(model.state_dict(),"checkpoints/mnist/rtest_%03d.pt" % epoch)
    

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
            t = torch.arange(100)
            t = t.type(torch.FloatTensor)
            t = t.to(device)

            data = data.to(device)

        
            data[:,:,14:16] = 0
            rx, w, phase= model(data, c, t)
            rdata = torch.ones_like(data)
            rdata[:,:,14:16] =0
            sample, _, _ = model(rdata * data, c, t)
            loss = loss_function(rx, data)
            test_loss += loss.item()
            if i == 0:
                c1 = torch.zeros_like(c)
                c1[:,6] = 1

                w1 = model.amplitude(c1)
                w1 = w1.view(data.shape[0], 30, 1)
                w1 = w1.repeat(1, 1, 100)
                
                x =  torch.sin(2 * np.pi * w1 * t + np.pi * phase)
                img = x
                x = x.sum(dim=1)
                x = model.d(x)
               
                n = min(data.size(0), 16)
                img1 = torch.cat((data[:32].view(-1,784), rx[:32], sample[:32], x[:32]),dim=0)
                img1 = img1.view(32 * 4, 1, 28, 28)
                img = img[:4]
                #img = img.view(4, 1, 50, 100)
                #save_image(img.cpu(),
                #         'images/_08_' + str(epoch) + '.png', nrow=1)
                save_image(img1.cpu(),
                         'images/rtest2_' + str(epoch) + '.png', nrow=32)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} '.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

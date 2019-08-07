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

class Layer(nn.Module):
    def __init__(self, c_size, in_size, out_size):
        super(Layer, self).__init__()
        self.main = nn.Linear(c_size , in_size * out_size, bias=False)
        self.b = nn.Linear(c_size, out_size)

        self.in_size = in_size
        self.out_size = out_size
    
    def forward(self, c):
        w = self.main(c)
        w = w.view(-1, self.in_size, self.out_size)
        b = self.b(c)
        b = b.view(-1, 1, self.out_size)

        return w, b
        

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.layer1 = Layer(10, 784, 400)
        self.layer2 = Layer(10, 400, 50)
        self.layer3 = Layer(10, 100, 400)
        self.layer4 = Layer(10, 400, 784)

        self.angular = nn.Sequential(
            nn.Linear(10, 50),
        )

    def transform(self, ang, phase, t):
        ang = ang.view(-1, 50, 1)
        phase = phase.view(-1, 50, 1)
        ang = ang.repeat(1, 1, 100)
        phase = phase.repeat(1, 1, 100)
        x = torch.sin(2 * np.pi * ang * t  + np.pi * phase )
        x = x.sum(dim=1)
        x = x.view(-1, 1, 100)
        noise = torch.randn_like(x)
        x = noise + x

        return x
       

    def forward(self, x, c, t):
        w1, b1 = self.layer1(c)
        w2, b2 = self.layer2(c)
        w3, b3 = self.layer3(c)
        w4, b4 = self.layer4(c)

        x = x.view(-1, 1, 784)
        x1 = torch.relu(torch.bmm(x, w1) + b1)

        ang = torch.sigmoid(self.angular(c))
        phase = torch.sigmoid(torch.bmm(x1, w2) + b2)

        x2 = self.transform(ang, phase, t)
        x3 = torch.relu(torch.bmm(x2, w3) + b3)
        x4 = torch.sigmoid(torch.bmm(x3, w4) + b4)

        return x4 

#class VAE(nn.Module):
#    def __init__(self):
#        super(VAE, self).__init__()
#        
#        self.e = Encoder()
#        self.d = Decoder()
#        self.amplitude = Key()
#
#    def forward(self, x, c, t):
#        x = x.view(-1, 784)
#        N = x.shape[0]
#
#        w = self.amplitude(c)
#        phase = self.e(x)
#
#        w = w.view(N, 50, 1)
#        phase = phase.view(N, 50, 1)
#       
#        w = w.repeat(1, 1, 100)
#        phase = phase.repeat(1, 1, 100)
#
#        x = torch.sin(2 * np.pi * w * t  + np.pi * phase )
#        x = x.sum(dim=1)
#        x = x.view(N, 100)         
#        noise = torch.randn_like(x)
#        x = noise + x
#        x = self.d(x)
#
#        return x, w, phase
#
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
        
        rx = model(data, c, t)
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
        torch.save(model.state_dict(),"checkpoints/mnist/cAE_%03d.pt" % epoch)
    

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
        
            rx = model(data, c, t)
            loss = loss_function(rx, data)
            test_loss += loss.item()
            if i == 0:
                c1 = torch.zeros_like(c)
                c1[:,6] = 1

                x = model(data, c1, t)
                
                input_data = torch.rand_like(data)
                x1 = model(input_data, c1, t)
               
                n = min(data.size(0), 16)
                img = torch.cat((data[:32].view(-1,784), rx[:32].view(-1,784), x[:32].view(-1, 784), x1[:32].view(-1, 784)),dim=0)
                img = img.view(32 * 4, 1, 28, 28)
                save_image(img.cpu(),
                         'images/cAE_05_' + str(epoch) + '.png', nrow=32)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} '.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

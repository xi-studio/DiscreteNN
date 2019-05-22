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
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
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

class Discrete(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x):
    substitute = torch.zeros_like(x) 
    index = x.argmax(dim=1)
    substitute.scatter_(1, index.unsqueeze(1), 1)
    
    return substitute 

  def backward(ctx, grad_output):

    return grad_output

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 40)
        self.fc22 = nn.Linear(400, 40)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        n = F.relu(self.fc21(x))
        s = torch.sigmoid(self.fc22(x))
        x = n * s
        x = x.view(-1,20,2)
        x = norm(x, dim=1)
       
        return x 

    def forward(self, x):
        z = self.encode(x.view(-1, 784))

        return z 

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(40, 400)
        self.fc3 = nn.Linear(400, 784)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1,40)))
        x = torch.sigmoid(self.fc3(x))

        return x 

emodel = Encoder().to(device)
dmodel = Decoder().to(device)
optimizer_e = optim.Adam(emodel.parameters(), lr=1e-3)
optimizer_d = optim.Adam(dmodel.parameters(), lr=1e-3)


def loss_function(recon_x, x, z, z_d):
    BCE1 = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE2 = torch.mean((z - z_d) ** 2)

    return BCE1, BCE2

def train(epoch):
    emodel.train()
    dmodel.train()
    train_loss = 0

    for idx, (data, c) in enumerate(train_loader):
        data = data.to(device)
        
        optimizer_e.zero_grad()
        optimizer_d.zero_grad()
        
        z = emodel(data)
        z_d = Discrete.apply(z) 
        rx = dmodel(z_d)

        loss_r, loss_z = loss_function(rx, data, z, z_d.detach())
        loss_mean = torch.mean((z**2 - 0.1) ** 2) 

        loss = loss_r + loss_z * 1000 + loss_mean * 1000 
        loss.backward()
        train_loss += loss_r.item()

        optimizer_e.step()
        optimizer_d.step()

        if idx % args.log_interval == 0:
            rand_z = torch.rand_like(z)
            rand_z_d = Discrete.apply(rand_z)

            rand_x = dmodel(rand_z_d)
            rz = emodel(rand_x)
            loss_randz = torch.mean((rz - rand_z_d.detach()) ** 2) * 1000
    
            print('==>loss_r:%0.3f' % loss_r.item(), '==>loss_z:%0.3f' % loss_z.item(),
                 '==>loss_randz: %0.3f' % loss_randz.item(),  '==>loss_mean: %0.3f' % loss_mean.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader),
                100. * idx / len(train_loader),
                loss_r.item() / len(data)))
            img = rand_x.view(-1,1,28,28)
            save_image(img.cpu(),'images/07_keni_sample_epoch_%d.png' % epoch, nrow=10)

#    if epoch % 10 == 0:
#        torch.save(model.state_dict(),"checkpoints/mnist/un_class_10_%03d.pt" % epoch)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / 60000))


def test(epoch):
    emodel.eval()
    dmodel.eval()
    test_loss = 0

    with torch.no_grad():
        for idx, (data, c) in enumerate(test_loader):
            data = data.to(device)
            z = emodel(data)
            z_d = Discrete.apply(z) 
            rx = dmodel(z_d)

            loss_r, loss_z= loss_function(rx, data, z, z_d.detach())
            loss = loss_r + loss_z  
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def sample(epoch, i=1):
    model.eval()
    with torch.no_grad():
        z = torch.zeros(10,10)
        for i in range(10):
            z[i,i] = 1
            
        z = z.to(device)
        rx = model.decode(z)
        img = rx.view(10,1,28,28)
        save_image(img.cpu(),'results/un_classify_10_epoch_%d.png' % epoch, nrow=10)


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

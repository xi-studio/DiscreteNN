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
        self.fc21 = nn.Linear(400, 10)
        self.fc22 = nn.Linear(400, 10)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        n = F.relu(self.fc21(x))
        s = torch.sigmoid(self.fc22(x))
        x = n * s
        x = norm(x, dim=1)
       
        return x 

    def forward(self, x):
        z = self.encode(x.view(-1, 784))

        return z 

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(1000, 400)
        self.fc3 = nn.Linear(400, 784)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 1000)))
        x = torch.sigmoid(self.fc3(x))

        return x 



def loss_function(recon_x, x, z, z_d):
    BCE1 = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE2 = torch.mean((z - z_d) ** 2)
    BCE3 = torch.mean((z**2 - 0.5) ** 2)

    return BCE1, BCE2+BCE3 * 1000

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
    train_c_data = torch.load("checkpoints/mnist/c_10_500_data.pt") 
    index = torch.randint(0,60000,(64,))
    c = train_c_data[index]
    c = c.to(device)
    print(c.shape)
    dmodel = Decoder().to(device)
    dmodel.load_state_dict(torch.load('checkpoints/mnist/tree_10_100_099.pt'))
    dmodel.eval()
    y = dmodel(c)

    s = torch.zeros(64*100,10)
    idx = torch.randint(0,10,(64*100,))
    s.scatter_(1, idx.unsqueeze(1), 1)
    sa = s.view(64,100,10)
    sa = sa.to(device)
    print('sa',sa.shape)
    
    su = train_c_data.sum(dim=0)
    d = (su>100).type(torch.FloatTensor)
    d = d.to(device)
    d = d.view(-1)
    
    print(d)
    mask = torch.zeros_like(c)  
    mask1 = torch.zeros_like(c)
    mask1[:,50:] = 1

    mask[:,:50] = 1
    with torch.no_grad():
        for i in range(100):
            condition = train_c_data[torch.randint(0,60000,(64,))]
            #y = dmodel(c * mask)
            y = dmodel(condition.to(device) * mask1 + sa * mask)
            #y = dmodel(sa * mask)
            y = y.view(64,1,28,28)
            save_image(y.cpu(),'images/sample_10_100_i_%d.png' % i, nrow=8)
            print(i)

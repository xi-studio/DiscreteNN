#
# Unsupervised classification of the entire Dataset into N categories
#
# Reconstruct the data and display the average image of each kind
#

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


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

class Discrete(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x):
    substitute = torch.zeros_like(x) 
    index = x.argmax(dim=1)
    substitute.scatter_(1, index.unsqueeze(1), 1)

    nonzero = x.sum(dim=1) > 0
    nonzero = nonzero.unsqueeze(1)
    nonzero = nonzero.type(torch.FloatTensor).to(device)
    
    return substitute * nonzero

  def backward(ctx, grad_output):

    return grad_output

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)


    def encode(self, x):
        x = F.relu(self.fc1(x))
        n = F.relu(self.fc21(x))
        s = torch.sigmoid(self.fc22(x))
        x = n * s
        x = x.view(-1, 10, 2)
        x = norm(x, dim=1)
       
        return x 

    def decode(self, z):
        x = F.relu(self.fc3(z))
        
        return torch.sigmoid(self.fc4(x))

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        z_d = Discrete.apply(z) 

        mask = torch.ones_like(z_d)
        mask[:64,:,1] = 0
        z_d = z_d * mask
        x = z_d.view(-1, 20)
        x = self.decode(x)

        return x, z, z_d


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, z, z_d):
    BCE1 = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE2 = torch.mean((z - z_d) ** 2)

    return BCE1, BCE2


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        data = data.to(device)
        mask = torch.zeros(2)
        
        optimizer.zero_grad()
        
        rx, z, z_d = model(data)
        loss_r, loss_z = loss_function(rx, data, z, z_d.detach())
        loss = loss_r + loss_z 
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('==>loss_r:', loss_r.item(), '==>loss_z:', loss_z.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

#    if epoch % 10 == 0:
#        torch.save(model.state_dict(),"checkpoints/mnist/dae_%03d.pt" % epoch)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            rx, z, z_d = model(data)
            loss_r, loss_z= loss_function(rx, data, z, z_d.detach())
            loss = loss_r + loss_z  
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def sample():
    model.eval()
    with torch.no_grad():
        z = torch.zeros(100,20)
        num = 0
        for i in range(10):
            for j in range(10):
                z[num,i] = 1
                z[num,10 +j] = 1
                num += 1
            
        z = z.to(device)
        rx = model.decode(z)
        img = rx.view(100,1,28,28)
        save_image(img.cpu(),'results/un_classify_100.png', nrow=10)



if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
    sample()

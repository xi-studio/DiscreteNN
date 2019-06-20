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

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 3)
        self.fc22 = nn.Linear(400, 3)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        n = F.relu(self.fc21(x))
        s = torch.sigmoid(self.fc22(x))
        x = n * s
        z = norm(x, dim=1)
        
        return z

    def forward(self, x):
        x = x.view(-1, 784)
        z = self.encode(x)
       
        return z

class Decoder(nn.Module):
    def __init__(self, c):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(c, 400)
        self.fc2 = nn.Linear(c, 400)
       
        self.fc3 = nn.Linear(400, 784)

    def forward(self, z):
        s = torch.sigmoid(self.fc1(z))
        n = F.relu(self.fc2(z))
        x = s * n
        x = torch.sigmoid(self.fc3(x))
       
        return x 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.e1 = Encoder()
        self.e2 = Encoder()
        self.e3 = Encoder()
        self.d1 = Decoder(13)
        self.d2 = Decoder(16)
        self.d3 = Decoder(19)

    def forward(self, x, c):
        z1 = self.e1(x)
        z2 = self.e2(x)
        z3 = self.e3(x)

        c1 = torch.cat((c, z1), dim=1)
        c2 = torch.cat((c, z1.detach(), z2), dim=1)
        c3 = torch.cat((c, z1.detach(), z2.detach(), z3), dim=1)

        x1 = self.d1(c1)
        x2 = self.d2(c2)
        x3 = self.d3(c3)

        return x1, x2, x3, z1, z2, z3


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
        c_onehot = c_onehot.to(device)

        data = data.to(device)
        optimizer.zero_grad()
        
        i = epoch % 3 + 1
        x1, x2, x3, z1, z2, z3 = model(data, c_onehot) 
        loss1 = loss_function(x1, data)
        loss2 = loss_function(x2, data)
        loss3 = loss_function(x3, data)
        loss1.backward()
        loss2.backward()
        loss3.backward()
        train_loss += loss1.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}, Loss2: {:.6f}, Loss3: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss1.item() / len(data), loss2.item() / len(data),loss3.item() / len(data)))

    if epoch % 50 == 0:
        torch.save(model.state_dict(),"checkpoints/mnist/share_%03d.pt" % epoch)
    

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
            c_onehot = c_onehot.to(device)
            data = data.to(device)
            x1, x2, x3, z1, z2, z3 = model(data, c_onehot)
            loss = loss_function(x3, data)
            test_loss += loss.item()
            if i == 0:
                c1 = torch.zeros_like(c_onehot)
                c1[:,6] = 1
                rz1 = torch.cat((c1, z1),dim=1)
                rz2 = torch.cat((c1, z1, z2),dim=1)
                rz3 = torch.cat((c1, z1, z2, z3),dim=1)
                
                rx1 = model.d1(rz1)
                rx2 = model.d2(rz2)
                rx3 = model.d3(rz3)
            
                n = min(data.size(0), 16)
                data = data.view(-1, 784)
                img = torch.cat((data[:64], x1[:64], x2[:64], x3[:64], rx1[:64], rx2[:64], rx3[:64]),dim=0)
                img = img.view(-1, 1, 28, 28)
                print(img.shape)
                save_image(img.cpu(),
                         'images/tree_02_sample_' + str(epoch) + '.png', nrow=64)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

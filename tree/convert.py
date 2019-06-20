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

        self.fc1 = GLU(784, 400)
        self.fc2 = GLU(400, 50)

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = norm(x, dim=1)

        return x 

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = GLU(50, 400)
        self.fc2 = nn.Linear(400, 784)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.sigmoid(self.fc2(x))

        return x 

class Transform(nn.Module):
    def __init__(self, c1, c2):
        super(Transform, self).__init__()

        self.fc1 = GLU(c1, c1)
        self.fc2 = GLU(c1, c1)
        self.fc3 = GLU(c1, c2)

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = norm(x, dim=1)

        return x 

class Key(nn.Module):
    def __init__(self, c1, c2):
        super(Key, self).__init__()

        self.fc1 = GLU(c1, c1)
        self.fc2 = GLU(c1, c1)
        self.fc3 = GLU(c1, c2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = norm(x, dim=1)

        return x 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.e = Encoder()
        self.d = Decoder()
        self.t = Transform(60, 50)
        self.k = Key(20, 50)


    def forward(self, x, c):
        x = x.view(-1, 784)

        z = self.e(x)
        z_t = torch.cat((z, c), dim=1)
        z_t = self.t(z_t)
         
        x = self.d(z_t)

        v = torch.ones_like(c)
        v = torch.cat((v, c), dim=1)
        z_k = self.k(v)
        
        similarity = (z_t * z_k).sum(dim=1)

        return x, z, similarity


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, s):
    target = torch.ones_like(s)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE1 = F.binary_cross_entropy(s, target, reduction='sum')

    return BCE, BCE1


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
        
        rx, z, s= model(data, c_onehot)
        loss, similarity = loss_function(rx, data, s)
        loss_a = loss + similarity
        loss_a.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data), 
                similarity.item() / len(data), 
                ))

    if epoch % 50 == 0:
        torch.save(model.state_dict(),"checkpoints/mnist/share_%03d.pt" % epoch)
    

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
            c_onehot = c_onehot.to(device)
            data = data.to(device)
            rx, z, s = model(data, c_onehot)
            loss, similarity = loss_function(rx, data, s)
            test_loss += loss.item()
            test_similarity += similarity.item()
            if i == 0:
                c1 = torch.zeros_like(c_onehot)
                c1[:,6] = 1
                z_t = torch.cat((z, c1), dim=1)
                z_t = model.t(z_t)
                x = model.d(z_t)

                n = min(data.size(0), 16)
                img = torch.cat((data[:64].view(-1,784), rx[:64], x[:64]),dim=0)
                img = img.view(64*3, 1, 28, 28)
                save_image(img.cpu(),
                         'images/z_' + str(epoch) + '.png', nrow=64)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}  {:.4f}'.format(test_loss, test_similarity))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

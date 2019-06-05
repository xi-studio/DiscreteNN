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
soft = nn.Softmax(dim=1)

class Discrete(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x):
    substitute = torch.zeros_like(x) 
    index = x.argmax(dim=1)
    substitute.scatter_(1, index.unsqueeze(1), 1)

    return substitute 

  def backward(ctx, grad_output):

    return grad_output


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 5)
        self.fc22 = nn.Linear(400, 5)
        self.fc3 = nn.Linear(15, 400)
        self.fc4 = nn.Linear(400, 784)


    def encode(self, x):
        x = F.relu(self.fc1(x))
        n = F.relu(self.fc21(x))
        s = torch.sigmoid(self.fc22(x))
        x = n * s
        x = x.view(-1, 5, 1)
        z = norm(x, dim=1)
        z_d = z
        #z_d = soft(z)
       
        return z, z_d 

    def decode(self, z):
        x = F.relu(self.fc3(z))
        
        return torch.sigmoid(self.fc4(x))

    def forward(self, x, c):
        z, z_d = self.encode(x.view(-1, 784))
        x = z_d.view(-1, 5)
        x = torch.cat((x, c),dim=1)
        x = self.decode(x)

        return x, z, z_d


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


mse = nn.MSELoss(reduction='sum')

def loss_function(recon_x, x, z, z_d):
    BCE1 = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    z_t = Discrete.apply(z_d)
    BCE2 = mse(z_d, z_t.detach())

    zf = z.flip([0])
    y = torch.sum((z*zf),dim=1)
    target = torch.ones_like(y)
    BCE3 = mse(y, target)

    return BCE1, BCE2, BCE3


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
        rx, z, z_d = model(data, c_onehot)
        loss_r, loss_z, loss_c = loss_function(rx, data, z, z_d.detach())
        loss = loss_r  #+ loss_c
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('==>loss_r:', loss_r.item(), '==>loss_z:', loss_z.item(),
                '==>loss_c:', loss_c.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    if epoch % 50 == 0:
        torch.save(model.state_dict(),"checkpoints/mnist/soft_%03d.pt" % epoch)
    

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


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
            rx, z, z_d = model(data, c_onehot)
            loss_r, loss_z, loss_c= loss_function(rx, data, z, z_d.detach())
            loss = loss_r + loss_z + loss_c  
            test_loss += loss_r.item()
            if i == 0:
                c1 = torch.zeros_like(c_onehot)
                c1[:,6] = 1
                x = z_d.view(-1, 5)
                x = torch.cat((x, c1),dim=1)
                x = model.decode(x)
                n = min(data.size(0), 16)
                img = torch.cat((data[:64].view(-1,784), rx[:64], x[:64]),dim=0)
                img = img.view(64*3, 1, 28, 28)
                save_image(img.cpu(),
                         'images/soft_11_sample_' + str(epoch) + '.png', nrow=64)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

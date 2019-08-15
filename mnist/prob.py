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

soft = nn.Softmax(dim=2)

class Discrete(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x):
    substitute = torch.zeros_like(x) 
    index = x.argmax(dim=2)
    substitute.scatter_(2, index.unsqueeze(2), 1)

    return substitute 

  def backward(ctx, grad_output):

    return grad_output


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(794, 400)
        self.fc2 = nn.Linear(400, 40 * 32)
        self.fc3 = nn.Linear(40 * 32 + 10, 400) 
        self.fc4 = nn.Linear(400, 784)

        self.tag = nn.Linear(10, 40 * 32)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 40, 32)
        x = soft(x)
       
        return x 

    def decode(self, z):
        x = F.relu(self.fc3(z))
        
        return torch.sigmoid(self.fc4(x))

    def forward(self, x, c):
        x = x.view(-1, 784)
        x = torch.cat((x, c), dim=1)
        z = self.encode(x)

        z1 = F.relu(self.tag(c))
        z1 = z1.view(-1, 40, 32)
        z1 = soft(z1)
        
        noise = torch.rand_like(z) * 0.5
        z = z + noise + z1
        z = soft(z)
        z_d = Discrete.apply(z) 

        x = z_d.view(-1, 40 * 32)
        x = torch.cat((x, c), dim=1)
        x = self.decode(x)

        return x


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

        rx = model(data, c_onehot)
        loss = loss_function(rx, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    if epoch % 20 == 0:
        torch.save(model.state_dict(),"checkpoints/mnist/prob_%03d.pt" % epoch)
    

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
        
            c1 = torch.zeros_like(c_onehot)
            c1[:,3] = 1
            data = data.to(device)
            rx = model(data, c_onehot)
            loss = loss_function(rx, data)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                rx1 = model(data, c1)
               
                data = data.view(-1, 1, 28, 28)
                rx = rx.view(-1, 1, 28, 28)
                rx1 = rx1.view(-1, 1, 28, 28)
                img = torch.cat((data, rx, rx1), dim = 2)
                save_image(img[:48].cpu(),
                         'images/prob_b_' + str(epoch) + '.png', nrow=48)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

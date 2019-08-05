from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


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

img, _ = torch.load('data/train.pt')
img = img.type(torch.FloatTensor)
img = img/255.0
img = img.view(60000, 784)
img = img.to(device)

z_rand = torch.rand(60000, 100).to(device)

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()

        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, 784)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

model = Gen().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    return BCE 

def sort(x):

    x = x.unsqueeze(1)
    x = x.repeat(1, 100, 1)

    idx = np.arange(60000)
    np.random.shuffle(idx)
    t = img[idx[:10000]]

    res = torch.sum((x - t) ** 2, dim=2)
    index = torch.argmin(res, dim=1)
    
   

    return y


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in range(600):
        z = torch.rand(args.batch_size, 100).to(device)
        optimizer.zero_grad()
        x = model(z)
        t = search(x)
   
        loss = loss_function(x, t)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * 1000, 1000,
                100. * batch_idx / 1000,
                loss.item() / 1000))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / 1000))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        with torch.no_grad():
            z = torch.rand(64, 100).to(device)
            sample = model(z)
            save_image(sample.view(64, 1, 28, 28),
                       'images/sample_gen_' + str(epoch) + '.png')

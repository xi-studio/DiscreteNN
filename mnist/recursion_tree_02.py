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

train_set = torch.load('data/train.pt')
test_set = torch.load('data/test.pt')

train_c_data = torch.zeros(60000,100,10)
train_data = train_set[0].type(torch.FloatTensor)
train_data = train_data/255.0

test_c_data = torch.zeros(10000,500)
test_data = test_set[0].type(torch.FloatTensor)
test_data = test_data/255.0

train_index = np.arange(60000)
np.random.shuffle(train_index)
train_index = torch.from_numpy(train_index)

test_index = np.arange(10000)
np.random.shuffle(test_index)
test_index = torch.from_numpy(test_index)


norm = torch.nn.functional.normalize

class Discrete(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x):
    substitute = torch.zeros_like(x) 
    index = x.argmax(dim=1)
    substitute.scatter_(1, index.unsqueeze(1), 1)

    #nonzero = x.sum(dim=1) > 0
    #nonzero = nonzero.unsqueeze(1)
    #nonzero = nonzero.type(torch.FloatTensor).to(device)
    
    return substitute 
    #* nonzero

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

def train(epoch, i, emodel, dmodel, optimizer_e, optimizer_d):
    emodel.train()
    dmodel.train()
    train_loss = 0

    k = int(60000 / args.batch_size)
    for idx in range(k):
        index = train_index[idx * args.batch_size : (idx + 1) * args.batch_size]
        data = train_data[index]
        c = train_c_data[index]
        data = data.to(device)
        c = c.to(device)
        
        optimizer_e.zero_grad()
        optimizer_d.zero_grad()
        
        z = emodel(data)
        z_d = Discrete.apply(z) 
        c[:,i] = z_d 
        rx = dmodel(c)

        loss_r, loss_z = loss_function(rx, data, z, z_d.detach())
        loss = loss_r + loss_z 
        loss.backward()
        train_loss += loss.item()

        optimizer_e.step()
        optimizer_d.step()

        if idx % args.log_interval == 0:
            print('==>loss_r:', loss_r.item(), '==>loss_z:', loss_z.item())
            print('C: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i,
                epoch, idx * len(data), 60000,
                100. * idx / 60000,
                loss.item() / len(data)))

#    if epoch % 10 == 0:
#        torch.save(model.state_dict(),"checkpoints/mnist/un_class_10_%03d.pt" % epoch)

    y = train_c_data[0]
#    print(y)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / 60000))


def test(epoch, i=1):
    emodel.eval()
    dmodel.eval()
    test_loss = 0

    mask = 0
    with torch.no_grad():
        for i, (data, index) in enumerate(test_loader):

            c = testdata[index]
            data = data.to(device)
            z = emodel(data)
            z = z * mask
            z_d = Discrete.apply(z) 
            rz = z_d + c
            rx = dmodel(rz)

            loss_r, loss_z= loss_function(rx, data, z, z_d.detach())
            loss = loss_r + loss_z  
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def fill(i=1):
    emodel.eval()

    k = int(60000 / args.batch_size)
    for idx in range(k):
        index = train_index[idx * args.batch_size : (idx + 1) * args.batch_size]
        data = train_data[index]
        c = train_c_data[index]
        data = data.to(device)
        c = c.to(device)
        
        z = emodel(data)
        z_d = Discrete.apply(z) 
        c[:, i] = z_d
        train_c_data[index] = c.detach().cpu()

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
    for i in range(100):
        emodel = Encoder().to(device)
        dmodel = Decoder().to(device)
        optimizer_e = optim.Adam(emodel.parameters(), lr=1e-3)
        optimizer_d = optim.Adam(dmodel.parameters(), lr=1e-3)
        for epoch in range(1, args.epochs + 1):
            train(epoch, i, emodel, dmodel, optimizer_e, optimizer_d)
        torch.save(dmodel.state_dict(),"checkpoints/mnist/tree_10_100_%03d.pt" % i)
        fill(i)
        zmean = train_c_data.sum(dim=0)
        print("Zmean:", zmean[i])
        save_image(zmean.cpu(),'results/tree_10_100_i_%d.png' % i)
    torch.save(train_c_data.cpu(), "checkpoints/mnist/c_10_500_data.pt")

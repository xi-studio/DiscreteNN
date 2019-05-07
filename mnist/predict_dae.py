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

def trans(x):
    index = x.argmax(dim=1) + 1
    nonzero = x.sum(dim=1) > 0
    index = index.type(torch.FloatTensor).to(device)
    nonzero = nonzero.type(torch.FloatTensor).to(device)
    return index * nonzero

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
        self.fc21 = nn.Linear(400, 10)
        self.fc22 = nn.Linear(400, 10)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)


    def encode(self, x):
        x = F.relu(self.fc1(x))
        n = F.relu(self.fc21(x))
        s = torch.sigmoid(self.fc22(x))
        x = n * s
        x = x.view(-1, 10, 1)
        x = norm(x, dim=1)
       
        return x 

    def decode(self, z):
        x = F.relu(self.fc3(z))
        
        return torch.sigmoid(self.fc4(x))

    def forward(self, x, c):
        z = self.encode(x.view(-1, 784))
        z_d = Discrete.apply(z) 
        x = z_d.view(-1, 10)
        x = torch.cat((x, c),dim=1)
        x = self.decode(x)

        return x, z, z_d


model = VAE()
model.load_state_dict(torch.load('checkpoints/mnist/dae_010.pt'))
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, z, z_d):
    BCE1 = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE2 = torch.mean((z - z_d) ** 2)

    return BCE1, BCE2

def mask():
    model.eval()
    with torch.no_grad():
        for i, (data, c) in enumerate(test_loader):
            c = c.unsqueeze(1)
            c_onehot = torch.zeros(data.shape[0], 10)
            c_onehot.scatter_(1, c, 1)
            c_onehot = c_onehot.to(device)
            noise = torch.rand_like(data)
            #data = data + data * noise 
            data[:,:,10:12] = 0

            data = data.to(device)
            rx, z, z_d = model(data)
             

            rx = rx.view_as(data)
            img = torch.cat((data,rx),dim=2)
            save_image(img[:8].cpu(), 'results_vae/test_%d.png' % i)
            print('i:',i)
            break

def test():
    model.eval()
    test_loss = 0
    
    z_mean = torch.zeros(10,2)
    z_mean = z_mean.to(device)
    num  = 0
    with torch.no_grad():
        for i, (data, c) in enumerate(test_loader):
            c = c.unsqueeze(1)
            c_onehot = torch.zeros(data.shape[0], 10)
            c_onehot.scatter_(1, c, 1)
            c_onehot = c_onehot.to(device)
            data = data.to(device)
            rx, z, z_d = model(data, c_onehot)
            loss_r, loss_z = loss_function(rx, data, z, z_d.detach())
            loss = loss_r + loss_z 
            test_loss += loss.item()

            z_mean += z_d.mean(dim=0)

            num += 1
            #print(num)

        print(z_mean/num)
            #img = rx[:64].view(64,1,28,28)
            #save_image(img.cpu(),
            #     'results/c1_x_%d.png' % i, nrow=8)

       
        #torch.save(zm.cpu(), './data/zmean/dis_c_%d.pt' % tag)

        # x = model.decode(Discrete.apply(zm).view(1,400))
       


def sample(tag):
    #c = torch.load('./data/zmean/dis_c_%d.pt' % tag)

    s = torch.rand(64, 10, 1) 
    s = norm(s,dim=1)
    s = s.to(device)
    z = Discrete.apply(s).view(-1,10)
    c = torch.zeros(64,10)
    c[:,tag] = 1
    c = c.to(device)
     
    print(z.shape,c.shape)

    z = torch.cat((z,c),dim=1)

    #z = z.to(device)
    
    model.eval()
    test_loss = 0

    x = model.decode(z)
    
    img = x.view(-1,1,28,28)
    save_image(img.cpu(), 'results/dis_sample_%d.png' % tag, nrow=8)

if __name__ == "__main__":
#    test()
#    mask()    
    for i in range(10):
#        test(i)
        sample(i)

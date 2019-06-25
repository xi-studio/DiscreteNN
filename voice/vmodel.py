import torch.nn as nn
import torch.nn.functional as F
import torch


class Bottleneck(nn.Module):
    def __init__(self, c1, c2):
        super(Bottleneck, self).__init__()
        self.h1 = nn.Sequential(
            nn.Conv1d(c1, c2, 3, 1, 1),
            nn.InstanceNorm1d(c2),
        ) 

        self.h1_gates = nn.Sequential(
            nn.Conv1d(c1, c2, 3, 1, 1),
            nn.InstanceNorm1d(c2),
        ) 

        self.h2 = nn.Sequential(
            nn.Conv1d(c2, c1, 3, 1, 1),
            nn.InstanceNorm1d(c1),
        ) 


    def forward(self, x):
        identity = x
        x = self.h1(x) * torch.sigmoid(self.h1_gates(x))
        out = self.h2(x)
        out += identity

        return out


class Encode(nn.Module):
    def __init__(self, DIM=128):
        super(Encode, self).__init__()

        self.h1 = nn.Conv1d(24, 128, 15, 1, 7)
        self.h1_gates = nn.Conv1d(24, 128, 15, 1, 7)

        self.h2 = nn.Sequential(
            nn.Conv1d(128, 256, 5, 2, 2),
            nn.InstanceNorm1d(256),
        ) 

        self.h2_gates = nn.Sequential(
            nn.Conv1d(128, 256, 5, 2, 2),
            nn.InstanceNorm1d(256),
        ) 

        self.h3 = nn.Sequential(
            nn.Conv1d(256, 512, 5, 2, 2),
            nn.InstanceNorm1d(512),
        ) 

        self.h3_gates = nn.Sequential(
            nn.Conv1d(256, 512, 5, 2, 2),
            nn.InstanceNorm1d(512),
        ) 

        reslayers = []
        for x in range(6):
            reslayers.append(Bottleneck(DIM * 4, DIM * 8))

        self.bottle = nn.Sequential(*reslayers)


    def forward(self, x):
        x = self.h1(x) * torch.sigmoid(self.h1_gates(x))
        x = self.h2(x) * torch.sigmoid(self.h2_gates(x))
        x = self.h3(x) * torch.sigmoid(self.h3_gates(x))

        x = self.bottle(x)


        return x


class Decode(nn.Module):
    def __init__(self, DIM=128):
        super(Decode, self).__init__()

        self.e0 = Layer(3, DIM)
        self.e1 = Layer(DIM, DIM * 2)
        self.e2 = Layer(DIM * 2, DIM * 4)
        self.conv = nn.Conv1d(DIM * 4, 24, 1)

        reslayers = []
        for x in range(6):
            reslayers.append(Bottleneck(DIM * 4, DIM * 4))

        self.bottle = nn.Sequential(*reslayers)

    def forward(self, x):
        x = self.e0(x)
        x = self.e1(x)
        x = self.e2(x)
        x = self.bottle(x)
        x = self.conv(x)

        return x


class VAE(nn.Module):
    def __init__(self, DIM=500):
        super(VAE, self).__init__()

        self.enc = Encode()

    def encode(self, x):
        x = self.enc(x)

        return x 


#    def decode(self, z):
#        x = self.dec(z)
#
#        return x


    def forward(self, x):
        x = self.enc(x)

        return x

class Key(nn.Module):
    def __init__(self):
        super(Key, self).__init__()

        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 50)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        w = torch.sigmoid(self.fc2(x))

        return w 

#class VAE(nn.Module):
#    def __init__(self):
#        super(VAE, self).__init__()
#        
#        self.e = Encoder()
#        self.d = Decoder()
#        self.amplitude = Key()
#
#    def forward(self, x, c, t):
#        N = x.shape[0]
#
#        w = self.amplitude(c)
#        phase = self.e(x)
#        px = torch.cat((phase, w), dim=1)
#        
#
#        w = w.view(N, 50, 1)
#        phase = phase.view(N, 50, 1)
#       
#        w = w.repeat(1, 1, 100)
#        phase = phase.repeat(1, 1, 100)
#
#        x = torch.sin(2 * np.pi * w * t  + np.pi * phase )
#        x = x.sum(dim=1)
#        x = x.view(N, 100)         
#        noise = torch.randn_like(x)
#        x = noise + x
#        
#       
#        x = self.d(px)
#
#        return x, w, phase

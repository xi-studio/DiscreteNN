import torch.nn as nn
import torch.nn.functional as F
import torch

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)


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


class Encoder(nn.Module):
    def __init__(self, c=24, DIM=64):
        super(Encoder, self).__init__()

        self.h1 = nn.Conv1d(c, DIM, 15, 1, 7)
        self.h1_gates = nn.Conv1d(c, DIM, 15, 1, 7)

        self.h2 = nn.Sequential(
            nn.Conv1d(DIM, DIM * 2, 5, 2, 2),
            #nn.InstanceNorm1d(DIM * 2),
        ) 

        self.h2_gates = nn.Sequential(
            nn.Conv1d(DIM, DIM * 2, 5, 2, 2),
            #nn.InstanceNorm1d(DIM * 2),
        ) 

        self.h3 = nn.Sequential(
            nn.Conv1d(DIM * 2, DIM * 4, 5, 2, 2),
            #nn.InstanceNorm1d(DIM * 4),
        ) 

        self.h3_gates = nn.Sequential(
            nn.Conv1d(DIM * 2, DIM * 4, 5, 2, 2),
            #nn.InstanceNorm1d(DIM * 4),
        ) 

        reslayers = []
        for x in range(3):
            reslayers.append(Bottleneck(DIM * 4, DIM * 8))

        self.bottle = nn.Sequential(*reslayers)


    def forward(self, x):
        x = self.h1(x) * torch.sigmoid(self.h1_gates(x))
        x = self.h2(x) * torch.sigmoid(self.h2_gates(x))
        x = self.h3(x) * torch.sigmoid(self.h3_gates(x))

        x = self.bottle(x)


        return x

class Decoder(nn.Module):
    def __init__(self, c=24, DIM=64):
        super(Decoder, self).__init__()

        self.h1 = nn.Sequential(
            nn.Conv1d(DIM * 4, DIM * 4, 5, 1, 2),
            PixelShuffle(upscale_factor=2),
            #nn.InstanceNorm1d(DIM * 2),
        ) 

        self.h1_gates = nn.Sequential(
            nn.Conv1d(DIM * 4, DIM * 4, 5, 1, 2),
            PixelShuffle(upscale_factor=2),
            #nn.InstanceNorm1d(DIM * 2),
        ) 

        self.h2 = nn.Sequential(
            nn.Conv1d(DIM * 2, DIM * 2, 5, 1, 2),
            PixelShuffle(upscale_factor=2),
            #nn.InstanceNorm1d(DIM),
        ) 

        self.h2_gates = nn.Sequential(
            nn.Conv1d(DIM * 2, DIM * 2 , 5, 1, 2),
            PixelShuffle(upscale_factor=2),
            #nn.InstanceNorm1d(DIM),
        ) 

        self.h3 = nn.Conv1d(DIM, c, 15, 1, 7)
        self.h3_gates = nn.Conv1d(DIM, c, 15, 1, 7)

        reslayers = []
        for x in range(3):
            reslayers.append(Bottleneck(DIM * 4, DIM * 8))

        self.bottle = nn.Sequential(*reslayers)


    def forward(self, x):
        x = self.bottle(x)
        x = self.h1(x) * torch.sigmoid(self.h1_gates(x))
        x = self.h2(x) * torch.sigmoid(self.h2_gates(x))
        x = self.h3(x) * torch.sigmoid(self.h3_gates(x))

        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.enc = Encoder()
        self.ef0 = Encoder(1, 20)
        self.dec = Decoder()


    def forward(self, x, f0):
        x = self.enc(x)
        zf0 = self.ef0(f0)
        x = self.dec(x)

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

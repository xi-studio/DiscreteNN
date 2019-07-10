import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class BasicBlock_0(nn.Module):
    def __init__(self, c):
        super(BasicBlock_0, self).__init__()

        self.t = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, 1),
            nn.Tanh(),
        )
        self.s = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, 1),
            nn.Sigmoid(),
        )
        
        self.end = nn.Conv1d(c, c, 1)


    def forward(self, x):
        identity = x
        out = self.t(x) * self.s(x)
        out += identity
        out = torch.relu(self.end(out))

        return x


class BasicBlock_1(nn.Module):
    def __init__(self, c):
        super(BasicBlock_1, self).__init__()
        reslayers = []
        for x in range(3):
            reslayers.append(BasicBlock_0(c))

        self.bottle = nn.Sequential(*reslayers)

        self.end = nn.Conv1d(c, c, 1) 

    def forward(self, x):
        identity = x
        out = self.bottle(x)
        out += identity
        out = torch.relu(self.end(out))

        return out

class BasicBlock_2(nn.Module):
    def __init__(self, c):
        super(BasicBlock_2, self).__init__()

        reslayers = []
        for x in range(3):
            reslayers.append(BasicBlock_1(c))

        self.bottle = nn.Sequential(*reslayers)

        self.end = nn.Conv1d(c, c, 1) 

    def forward(self, x):
        identity = x
        out = self.bottle(x)
        out += identity
        out = torch.relu(self.end(out))

        return out

class BasicBlock_3(nn.Module):
    def __init__(self, c):
        super(BasicBlock_3, self).__init__()

        reslayers = []
        for x in range(3):
            reslayers.append(BasicBlock_2(c))

        self.bottle = nn.Sequential(*reslayers)

        self.end = nn.Conv1d(c, c, 1) 

    def forward(self, x):
        identity = x
        out = self.bottle(x)
        out += identity
        out = torch.relu(self.end(out))

        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.start = nn.Conv1d(24, 256, 1)
        self.conv1 = BasicBlock_3(256)
        self.conv2 = nn.Conv1d(256, 20, 1)

    def forward(self, x):
        x = self.start(x)
        x = self.conv1(x)
        x = self.conv2(x)
        phase = torch.sigmoid(x)

        return phase

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.start = nn.Conv1d(100, 256, 1)
        self.conv1 = BasicBlock_3(256)
        self.conv2 = nn.Conv1d(256, 24, 1)

    def forward(self, x):
        x = self.start(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x 

class Key(nn.Module):
    def __init__(self):
        super(Key, self).__init__()
        self.start = nn.Conv1d(3, 256, 1)
        self.conv1 = BasicBlock_3(256)
        self.conv2 = nn.Conv1d(256, 20, 1)

    def forward(self, x):
        x = self.start(x)
        x = self.conv1(x)
        x = self.conv2(x)
        w = torch.sigmoid(x)

        return w 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.w = Key()
        self.en = Encoder()
        self.de = Decoder()
       

    def forward(self, x, z, t):
        w = self.w(z)
        phase = self.en(x)

        phase = phase.unsqueeze(2)
        phase = phase.repeat(1, 1, 100, 1)

        w = w.unsqueeze(2)
        w = w.repeat(1, 1, 100, 1)
        w = w.transpose(2, 3).contiguous()
        w = w * t
        w = w.transpose(2, 3).contiguous()

        wav = torch.sin(2 * np.pi * w + np.pi * phase)
        wav = wav.sum(dim=1)
        noise = torch.randn_like(wav) * 10
        wav = noise + wav
        
        x = self.de(wav)

        return x 

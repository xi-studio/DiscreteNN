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
        self.start = nn.Conv1d(128, 256, 1)
        self.conv1 = BasicBlock_2(256)
        self.conv2 = BasicBlock_2(256)
        self.conv3 = BasicBlock_3(512)
        self.end   = nn.Conv1d(512, 20, 1)
        self.down1 = nn.Conv1d(256, 256, 4, 2, 1)
        self.down2 = nn.Conv1d(256, 512, 4, 2, 1)

    def forward(self, x):
        x = self.start(x)
        x = self.conv1(x)
        x = self.down1(x)
        x = self.conv2(x)
        x = self.down2(x)
        x = self.conv3(x)
        x = self.end(x)
        phase = torch.sigmoid(x)

        return phase

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.start = nn.Conv1d(100, 1024, 1)
        self.conv1 = BasicBlock_2(1024)
        self.conv2 = BasicBlock_2(512)
        self.conv3 = BasicBlock_2(256)
        self.end   = nn.Conv1d(256, 128, 1)
        self.up1   = nn.ConvTranspose1d(1024, 512, 4, 2, 1)
        self.up2   = nn.ConvTranspose1d(512, 256, 4, 2, 1)

    def forward(self, x):
        x = self.start(x)
        x = self.conv1(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.end(x)

        return x 

class Key(nn.Module):
    def __init__(self):
        super(Key, self).__init__()
        self.start = nn.Conv1d(2, 256, 1)
        self.conv1 = BasicBlock_2(256)
        self.conv2 = BasicBlock_2(256)
        self.conv3 = BasicBlock_3(256)
        self.end   = nn.Conv1d(256, 20, 1)
        self.down1 = nn.Conv1d(256, 256, 4, 2, 1)
        self.down2 = nn.Conv1d(256, 256, 4, 2, 1)

    def forward(self, x):
        x = self.start(x)
        x = self.conv1(x)
        x = self.down1(x)
        x = self.conv2(x)
        x = self.down2(x)
        x = self.conv3(x)
        x = self.end(x)
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
        noise = torch.randn_like(wav) 
        wav = noise + wav
        
        x = self.de(wav)

        return x 

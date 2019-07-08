import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, c):
        super(BasicBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, 1),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, 3, 1, 1),
            nn.BatchNorm1d(c),
        )


    def forward(self, x):
        identity = x
        out = self.main(x)
        out += identity

        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(256, 256, 1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, 15, 1, 7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, 5, 2, 2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(512, 1024, 5, 2, 2),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )


        reslayers = []
        for x in range(3):
            reslayers.append(BasicBlock(1024))

        self.bottle = nn.Sequential(*reslayers)

        self.w = nn.Parameter(data=torch.Tensor(1024,100),requires_grad=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bottle(x)

        x = x * self.w
        x = x.sum(dim=2)
        
        phase = torch.sigmoid(x)

        return phase

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv1d(256, 256, 1)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 4, 2, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 4, 2, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )


        reslayers = []
        for x in range(3):
            reslayers.append(BasicBlock(1024))

        self.bottle = nn.Sequential(*reslayers)

        self.fc = nn.Linear(1024, 1024 * 100)


    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, 100)
        x = self.bottle(x)
        
        x = self.conv4(x)
        x = self.conv3(x)
        x = self.conv2(x)
        x = self.conv1(x)
        
        return x 


class Key(nn.Module):
    def __init__(self):
        super(Key, self).__init__()

        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        w = torch.sigmoid(self.fc2(x))

        return w 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.amplitude = Key()

        self.en = nn.Sequential(
                    nn.Linear(240, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 300),
                    nn.Sigmoid(),
        )

        self.de = nn.Sequential(
                    nn.Linear(300, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 240),
        )

    def forward(self, x, c, t):
        x = self.en(x)
        x = self.de(x)
#        N = x.shape[0]
#        w = self.amplitude(c)
#        phase = self.en(x)
#        
#        w = w.view(N, 100, 1)
#        phase = phase.view(N, 100, 1)
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
#        x = self.de(x)
#        x = x.view(-1, 256, 400)

        return x 

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class GLU(nn.Module):
    def __init__(self, c1, c2):
        super(GLU, self).__init__()
        self.s = nn.Linear(c1, c2)
        self.g = nn.Linear(c1, c2)

    def forward(self, x):
        s = torch.sigmoid(self.s(x))
        g = torch.relu(self.g(x))
        output = s * g

        return output 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(800, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(400, 50)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        phase = torch.sigmoid(self.fc3(x))

        return phase

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = GLU(100, 400)
        self.fc2 = nn.Linear(400, 800)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.tanh(self.fc2(x))

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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.e = Encoder()
        self.d = Decoder()
        self.amplitude = Key()

        self.main = nn.Sequential(
                    nn.Conv1d(256,256,3,1,1),
                    nn.ReLU(),
                    nn.Conv1d(256,256,3,1,1),
        )

    def forward(self, x, c, t):
        x = self.main(x)
        return x
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

#        return x, w, phase

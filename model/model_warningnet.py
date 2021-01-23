import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from backbone import resnet

class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=0.5)
        dtype = torch.FloatTensor

    def forward(self, input, hidden):
        c1 = torch.cat((input, hidden), 1)
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = self.dropout(F.sigmoid(rt))
        update_gate = self.dropout(F.sigmoid(ut))
        gated_hidden = torch.mul(reset_gate, hidden)
        ct = F.tanh(input)

        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h


class WarningNet(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self):
        super(WarningNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.LeakyReLU(0.01))

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.LeakyReLU(0.01))

        self.rnn3 = GRUCell(input_size=32, hidden_size=32)

        self.conv41 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.lin1 = nn.Linear(150,1)
        self.lin2 = nn.Linear(150,1)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, px, cx):

        a1 = self.conv1(px)
        b1 = self.conv2(a1)

        a2 = self.conv1(cx)
        b2 = self.conv2(a2)

        c2 = self.rnn3(b2, b1)

        out_src = self.conv41(c2)
        out_src = torch.squeeze(self.lin1(out_src),-1)
        out_src = self.dropout(F.relu(out_src))
        out_src = torch.squeeze(self.lin2(out_src),-1)
        out_src = self.dropout(F.relu(out_src))
        out_src = torch.clamp(out_src, 0.0, 1.0)

        return out_src

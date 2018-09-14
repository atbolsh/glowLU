"""
Module to compute the nonlinearity used.
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from copy import deepcopy
import math
import numpy as np

import os


#alpha = 0.33**(1.0/3)
#beta  = 0.01**(1.0/3)

#r = beta / (3*alpha)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

"""
A = 20200./10000
B = 201./10000
C = 20201./10000
"""

A = 3./5
B = 4./5
C = 1

def elSelect(a, b, p):
    """a if p > 0, b if p < 0"""
    s = (torch.sign(p)+1)/2.
    return a*s + b*(1-s)

#func = nn.LeakyReLU(0.5)
########

def inv(y):
    return (torch.sqrt((torch.abs(y)+C)**2 - A**2) - B)*torch.sign(y)

def derInv(y):
    a = torch.abs(y)
    return (a + C)/torch.sqrt((a + C)**2 - A**2)


class nLayer(nn.Module):
    def __init__(self, n, bias=True):
        super(nLayer, self).__init__()
        self.n = n
        self.A2 = Parameter(torch.tensor(9./25))
        self.A2.data.uniform_(5./25, 15./25)
        self.B = Parameter(torch.tensor(4./5))
        self.B.data.uniform_(3./5, 1.)

        self.soft = torch.nn.Softmax()
    
    def func(self, x):
        B1 = self.soft(self.B - 1e-2) + 1e-2
        B  = 10 - self.soft(10 - B1)
        C = torch.sqrt(torch.abs(self.A2) + B**2)
        if self.A2 > 0:
            return (torch.sqrt((torch.abs(x) + B)**2 + self.A2) - C)*torch.sign(x)
        else:
            return (torch.sqrt((torch.abs(x)+C)**2 + self.A2) - B)*torch.sign(x)
 
    def der(self, x):
        """The derivative"""
        a = torch.abs(x)
        #Differentiable clamping, to avoid nan's
        B1 = self.soft(self.B - 1e-2) + 1e-2
        B  = 10 - self.soft(10 - B1)
        C = torch.sqrt(torch.abs(self.A2) + B**2)
        if self.A2 > 0:
            return (a + B)/torch.sqrt((a + B)**2 + self.A2)
        else:
            return (a + C)/torch.sqrt((a + C)**2 + self.A2)

    def inv(self, y):
        B1 = self.soft(self.B - 1e-2) + 1e-2
        B  = 10 - self.soft(10 - B1)
        C = torch.sqrt(torch.abs(self.A2) + B**2)
        if self.A2 > 0:
            return (torch.sqrt((torch.abs(y)+C)**2 - self.A2) - B)*torch.sign(y)
        else:
            return (torch.sqrt((torch.abs(y)+B)**2 - self.A2) - C)*torch.sign(y)
    
    def derInv(self, y):
        a = torch.abs(x)
        B1 = self.soft(self.B - 1e-2) + 1e-2
        B  = 10 - self.soft(10 - B1)
        C = torch.sqrt(torch.abs(self.A2) + B**2)
        if self.A2 < 0:
            return (a + B)/torch.sqrt((a + B)**2 - self.A2)
        else:
            return (a + C)/torch.sqrt((a + C)**2 - self.A2)
    
    def forward(self, x):
        return self.func(x), torch.sum(torch.log(torch.abs(self.der(x))), 1)
    
    def pushback(self, y):
        x = self.inv(y.detach())
        return x, torch.log(self.der(x))


"""
class backLayer(nn.Module):
    def __init__(self, n, bias=True):
        super(backLayer, self).__init__()
        self.n = n
    
    def forward(self, x):
        y = inv(x)
        return y, torch.sum(torch.log(torch.abs(derInv(x))), 1)#0-torch.sum(torch.log(torch.abs(der(y))), 1)
    
    def pushback(self, y):
        x = func(y.detach())
        return x, 0-torch.log(der(y))
"""
  



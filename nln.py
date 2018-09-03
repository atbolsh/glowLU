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


A = 20200./10000
B = 201./10000
C = 20201./10000

def elSelect(a, b, p):
    """a if p > 0, b if p < 0"""
    s = (torch.sign(p)+1)/2.
    return a*s + b*(1-s)

#func = nn.LeakyReLU(0.5)

def func(x):
    return (torch.sqrt((torch.abs(x) + B)**2 + A**2) - C)*torch.sign(x)
 
def der(x):
    """The derivative"""
    a = torch.abs(x)
    return (a + B)/torch.sqrt((a + B)**2 + A**2)

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
    
    def forward(self, x):
        return func(x), torch.sum(torch.log(torch.abs(der(x))), 1)
    
    def pushback(self, y):
        x = inv(y.detach())
        return x, torch.log(der(x))



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

  



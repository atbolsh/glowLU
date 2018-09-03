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

def elSelect(a, b, p):
    """a if p > 0, b if p < 0"""
    s = (torch.sign(p)+1)/2.
    return a*s + b*(1-s)

#func = nn.LeakyReLU(0.5)

def func(x):
    return elSelect(torch.sign(x)*(torch.abs(x)-2.0/3), 0.99*(x.pow(3)/3) + 0.01*x/3, torch.abs(x) - 1)

def der(x):
    """The derivative"""
    return elSelect(1, 0.99*x*x +0.01, torch.abs(x) - 1)

########

def cr(x):
    return torch.sign(x)*(torch.abs(x).pow(1.0/3))

def inv_middle(y):
    """Solution to 0.99*x**3/3 + 0.01x - y = 0 by cubic formula """
#    p = 0
#    Method is a little obtuse.
#    First, we set x = 100*y + z
#    Solving for z, in original formula and subbing in u = z**1/3, we get
#    0.33**1/3 u**3 + 0.01**1/3u + 100*0.33**1/3*y = 0
#
#    These substitutions make the formula more numerically stable.
     

    q  = y/(0.66)
    r  = 0.01/0.99
    
    b  =  torch.sqrt(q*q + r*r*r)
    return  cr(q + b) + cr(q - b)

#    return 100*y + u*u*u

def inv(y):
    return elSelect(torch.sign(y)*(torch.abs(y) + 2.0/3), inv_middle(y), torch.abs(3*y) - 1)
 
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
        return y, 0-torch.sum(torch.log(torch.abs(der(y))), 1)
    
    def pushback(self, y):
        x = func(y.detach())
        return x, 0-torch.log(der(y))

  



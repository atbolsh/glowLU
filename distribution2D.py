import nln
#from LU import *

#from sklearn.datasets import load_boston

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

import numpy as np
import os
import math

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
"""
#Not sure if this is the correct one, but here goes nothing.
def Cosine_sample(n):
    pre     = torch.Tensor(n, 2).uniform_(0-math.pi, math.pi).cuda()
    post    = torch.Tensor(n, 2).cuda()
    post[:, 0] = torch.cos(pre[:, 0] + pre[:, 1])
    post[:, 1] = torch.cos(pre[:, 0] - pre[:, 1])
    return post

#Compute the (unknown) p-value of this sample.
#Derived by hand
def Cosine_val(sample):
    a = torch.sin(torch.acos(sample))
    return (1/(2*math.pi**2))*(1/(a[:, 0]*a[:, 1]))
    
"""

def preColumn(n):
    a = torch.Tensor(n).cuda()
    a.uniform_(-2.3, -1.7)
    b = torch.randn(n).cuda()*0.2
    d = torch.Tensor(n).cuda()
    d.uniform_(0, 1)
    p = a*(((0<=d)*(d<0.2)).float()) + \
        (b - 1)*(((0.2<=d)*(d<0.4)).float()) + \
        b*(((0.4<=d)*(d<0.6)).float()) + \
        (b + 1)*(((0.6 <= d)*(d< 0.8)).float()) + \
        (a + 4)*((0.8<=d).float())
    return p

def Column(n):
    r = torch.Tensor(n, 2).cuda()
    r[:, 0] = preColumn(n)
    r[:, 1] = preColumn(n)
    return r

def columnVals(sample):
    a = sample[:, 0]
    b = sample[:, 1]
    s = math.sqrt(2*math.pi)
    p1 = (0.2/0.6)*(((a < -1.7)*(a > -2.3)).float()) + \
         (0.2/0.6)*(((a > 1.7)*(a < 2.3)).float()) + \
         0.2*(1/(0.2*s))*torch.exp(-a*a/(2*0.04)) + \
         0.2*(1/(0.2*s))*torch.exp(-(a-1)*(a-1)/(2*0.04)) + \
         0.2*(1/(0.2*s))*torch.exp(-(a+1)*(a+1)/(2*0.04)) 
    p2 = (0.2/0.6)*(((b < -1.7)*(b > -2.3)).float()) + \
         (0.2/0.6)*(((b > 1.7)*(b < 2.3)).float()) + \
         0.2*(1/(0.2*s))*torch.exp(-b*b/(2*0.04)) + \
         0.2*(1/(0.2*s))*torch.exp(-(b-1)*(b-1)/(2*0.04)) + \
         0.2*(1/(0.2*s))*torch.exp(-(b+1)*(b+1)/(2*0.04)) 
    return p1*p2

def normal(n):
    return torch.randn(n, 2).cuda()

def pvals(sample):
    return torch.exp(0-torch.sum(sample*sample, 1)/2)/(2*math.pi)



"""
Model of 2D Columns distribution.
"""

import distribution2D as d2

from nln import nLayer
from LU  import *

from sklearn.datasets import load_boston

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


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"



#Build the network

class layer1(nn.Module):
    def __init__(self):
        super(layer1, self).__init__()
        self.fc = LU(2)

    def forward(self, x):
        y, lJ = self.fc(x)
        return y, math.log(1/(2*math.pi)) - torch.sum(y*y, 1)/2 + lJ

    def sample(self, n, x = None):
        if (type(x) == type(None)):
            x = Variable(torch.randn(n, 2)).cuda()
        y, _ = self.fc.pushback(x)
        return y


batchsize = 10000
epochnum = 2


L = torch.nn.MSELoss()

target = torch.zeros(batchsize, 2).cuda()


#alpha=10

model = layer1().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

if __name__ == "__main__":
    for epoch in range(epochnum):
        xu = Variable(torch.randn(batchsize, 2)).cuda()
#        print(model.fc.weight.data)
        print(model.fc.logJ())
        print(model.fc.bias.data)
  
    
        xd = model.sample(batchsize)
#        print(xd)
    
        model.train()
        model.zero_grad()    
        yu, lu = model(xu)
#        print(xu)
#        print(yu)
        print((math.log(2*math.pi) + lu + 0.5*torch.sum(yu*yu, 1))[:10])
        yd, ld = model(xd)
        pu = torch.exp(lu)
        pd = torch.exp(ld)
#        print(pd)
        
#        loss = L(yu, target)#/batchsize # + torch.sum(ld)
        loss = 0 - torch.sum(lu) + torch.sum(ld)

#        print(loss)
#        print(torch.max(torch.sum(yu*yu, 1)))
        loss.backward()
        optimizer.step()
        
        model.eval()
        
        pt = d2.pvals(xu)
        
        print(epoch)
        print(torch.min(pt))
#        print(torch.max(torch.log(pt/pu)))
#        print(torch.min(torch.log(pt/pu)))
        print('\n')
        print('\n\n\n')



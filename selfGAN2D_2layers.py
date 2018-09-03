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
        self.fc1 = LU(2)
        self.nln = nLayer(2)
        self.fc2 = LU(2)
 
    def forward(self, x):
        y1, lJ1 = self.fc1(x)
        y2, lJ2 = self.nln(y1)
        y3, lJ3 = self.fc2(y2)
#        print(lJ2)
#        y = x + self.fc.bias
        return y3, math.log(1/(2*math.pi)) - torch.sum(y3*y3, 1)/2 + lJ1 + lJ2 + lJ3

    def sample(self, n, y3 = None):
        if (type(y3) == type(None)):
            y3 = Variable(torch.randn(n, 2)).cuda()
        y2, _ = self.fc2.pushback(y3)
        y1, _ = self.nln.pushback(y2) 
        x, _  = self.fc1.pushback(y1)
        return x


batchsize = 10000
epochnum = 20000


L = torch.nn.MSELoss()

target = torch.zeros(batchsize, 2).cuda()


#alpha=10

model = layer1().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

if __name__ == "__main__":
    for epoch in range(epochnum):
        xu = Variable(torch.randn(batchsize, 2)).cuda()
#        print(model.fc.weight.data)
        print(model.fc1.logJ())
        print(model.fc1.bias.data)
        print(model.fc2.logJ())
        print(model.fc2.bias.data)
  
    
        xd = model.sample(batchsize)
#        print(xd[:10])
    
        model.train()
        model.zero_grad()    
        yu, lu = model(xu)
        yd, ld = model(xd)

#        print(xu)
#        print(yu)
        print((math.log(2*math.pi) + lu + 0.5*torch.sum(yu*yu, 1))[:10])
        print((math.log(2*math.pi) + ld + 0.5*torch.sum(yd*yd, 1))[:10])
#        print(ld[:10])

        pu = torch.exp(lu)
        pd = torch.exp(ld)
#        print(pd)
        
#        loss = L(yu, target)#/batchsize # + torch.sum(ld)
        loss = 0 - torch.sum(lu) + torch.sum(ld)
        
        print('loss = ' + str(loss))
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



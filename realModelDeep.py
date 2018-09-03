"""
Model of 2D Columns distribution.
"""

import distribution2D as d2

from nln import nLayer
from nln import backLayer
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

class flowGAN(nn.Module):
    def __init__(self, n):
        """Even n preferred"""
        super(flowGAN, self).__init__()
        self.lin = nn.ModuleList([LU(2) for i in range(n+1)])
        hyp = []
        for i in range(n):
            if i % 2 == 0:
                hyp.append(nLayer(2))
            else:
                hyp.append(nLayer(2))
#                hyp.append(backLayer(2))
        self.hyp = nn.ModuleList(hyp)
        self.n = n
 
    def forward(self, x):
        y, lJ = self.lin[0](x)
        for i in range(self.n):
            y, nlJ = self.hyp[i](100*y) #Make it skinny
            y = y/100
            lJ = nlJ + lJ
            y, nlJ = self.lin[i+1](y)
            lJ = lJ + nlJ
        return y, math.log(1/(2*math.pi)) - torch.sum(y*y, 1)/2 + lJ

    def sample(self, n, x = None):
        if (type(x) == type(None)):
            x = Variable(torch.randn(n, 2)).cuda()
        x, _ = self.lin[-1].pushback(x)
        for i in range(i, self.n):
            x, _ = self.hyp[-i].pushback(100*x)/100
            x, _ = self.lin[-1-i].pushback(x)
        return x




batchsize = 10000
epochnum = 20000000


#alpha=10

model = flowGAN(200).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    for epoch in range(epochnum):
        xu = Variable(d2.Column(batchsize)).cuda()
#        print(model.fc.weight.data)
#        print(model.fc1.logJ())
#        print(model.fc1.bias.data)
#        print(model.fc2.logJ())
#        print(model.fc2.bias.data)
   
    
#        xd = model.sample(batchsize)
#        print(xd)
    
        model.train()
        model.zero_grad()    
        yu, lu = model(xu)
#        print(xu[:10])
        print(yu[:10])
        print((math.log(2*math.pi) + lu + 0.5*torch.sum(yu*yu, 1))[:10])
#        yd, ld = model(xd)
        pu = torch.exp(lu)
#        pd = torch.exp(ld)
#        print(pd)
        
#        loss = L(yu, target)#/batchsize # + torch.sum(ld)
        loss = 0 - torch.sum(lu) #/batchsize # + torch.sum(ld)

        print('loss = ' + str(loss))
        loss.backward()
        
        optimizer.step()
        
        model.eval()
        
        pt = d2.columnVals(xu)
        print(pt[:5])
        print(pu[:5])
        
        print(epoch)
        print('\n')
        print('\n\n\n')


        if epoch%100 == 0:
            torch.save(model, 'deepWeights/epoch' + str(epoch))





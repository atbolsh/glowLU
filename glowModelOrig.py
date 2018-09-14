"""
Model of 2D Columns distribution.

Uses GLOW.
"""

import distribution2D as d2

from nln import nLayer
#from nln import backLayer
from LU  import *
from affine2Dorig import *

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
os.environ["CUDA_VISIBLE_DEVICES"]="6"



#Build the network

class flowGAN(nn.Module):
    def __init__(self, n):
        """Even n preferred"""
        super(flowGAN, self).__init__()
        self.lin = nn.ModuleList([LU(2) for i in range(n+1)])
        aff = []
        for i in range(n):
            if i % 2 == 0:
                aff.append(affine2(2, hidden=128))
            else:
                aff.append(affine2(2, hidden=128))
#                aff.append(backLayer(2))
        self.aff = nn.ModuleList(aff)
        self.n = n
 
    def forward(self, x):
        y, lJ = self.lin[0](x)
        for i in range(self.n):
            y, nlJ = self.aff[i](y)
            lJ = nlJ + lJ
            y, nlJ = self.lin[i+1](y)
            lJ = lJ + nlJ
        return y, math.log(1/(2*math.pi)) - torch.sum(y*y, 1)/2 + lJ

    def sample(self, n, x = None):
        if (type(x) == type(None)):
            x = Variable(torch.randn(n, 2)).cuda()
        x, _ = self.lin[-1].pushback(x)
        for i in range(1, self.n+1):
            x, _ = self.aff[-i].pushback(x)
            x, _ = self.lin[-1-i].pushback(x)
        return x




batchsize = 1000
epochnum = 20000


#alpha=10

model = flowGAN(12).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    trainingCurve = open('glowOrig/curve', 'w')
    trainingCurve.write('Epoch\t\tLoss\n')
    trainingCurve.close()  
    for epoch in range(epochnum):
        xu = Variable(d2.Column(batchsize)).cuda()
#        print(model.fc.weight.data)
#        print(model.fc1.logJ())
#        print(model.fc1.bias.data)
#        print(model.fc2.logJ())
#        print(model.fc2.bias.data)
   
    
        xd = model.sample(batchsize)
#        print(xd)
    
        model.train()
        model.zero_grad()    
        yu, lu = model(xu)
#        print(xu[:10])
        print(yu[:10])
        print((math.log(2*math.pi) + lu + 0.5*torch.sum(yu*yu, 1))[:10])
        yd, ld = model(xd.detach())
        pu = torch.exp(lu)
#        pd = torch.exp(ld)
#        print(pd)
        
#        loss = L(yu, target)#/batchsize # + torch.sum(ld)
        loss = 0 - torch.sum(lu)# + torch.sum(ld)

        loss.backward()
        
        optimizer.step()
        
        model.eval()
        yu, lu = model(xu)
#        print('adversarial dif = ' + str(torch.sum(ld) - torch.sum(lu)))
       
        pt = d2.columnVals(xu)
        print(pt[:5])
        print(pu[:5])
        
        print(epoch)
#        print(model.aff[0].b.weight)
        print('\n')
        print('\n\n\n')

        if epoch%100 == 0:
            torch.save(model, 'glowOrig/epoch' + str(epoch))
            trainingCurve = open('glowOrig/curve', 'a')
            trainingCurve.write(str(epoch) + '\t\t' + str(loss) + '\n')
            trainingCurve.close()









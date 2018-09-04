import torch

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
os.environ["CUDA_VISIBLE_DEVICES"]="2"



class event(nn.Module):
    def __init__(self):
        """Even n preferred"""
        super(event, self).__init__()
        self.q = Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        return torch.log(self.q)*x + torch.log(1 - self.q)*(1 - x)
    
    def sample(self, n):
        x = torch.zeros(n).cuda()
        x.uniform_(0, 1)
        return (torch.sign(x + self.q -1) + 1)/2.


model = event().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(10000):
    xu = torch.zeros(1000).cuda()
    xu.uniform_(0, 1)
    xu = (torch.sign(xu -0.1) + 1)/2

    xd = model.sample(1000)

    model.train()
    model.zero_grad()    
 
    L = torch.sum(model(xd)) - torch.sum(model(xu))

    L.backward()
    optimizer.step()
    print(epoch)
    print(L)
    print(model.q)
    print('\n\n\n') 







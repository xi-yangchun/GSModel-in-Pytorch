import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gsmodel import GSCott
from monitor import Monitor
from multiprocessing import Process

#a=torch.tensor(np.random.rand(1,1,200,200)).float()
u=torch.tensor(np.ones((1,1,256,256))).float()
u[0,0,118:138,118:138]=0.5
v=torch.tensor(np.zeros((1,1,256,256))).float()
v[0,0,118:138,118:138]=0.25
R=100
T=1
Du = 2e-5
Dv = 1e-5
f = 0.022
k = 0.051
gs=GSCott(u,v,Du,Dv,f,k,R,T)

m=Monitor()
m.run(gs)
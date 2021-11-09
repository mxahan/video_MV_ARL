# Import libraries 
import os

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch 

import pickle 
import matplotlib.pyplot as plt

import numpy as np

import cv2

import  time
import glob

from scipy.io import loadmat

import random

from random import seed, randint

from sklearn.model_selection import train_test_split

import pandas as pd

from imutils.video import FPS

import imutils

from torch.utils.data import DataLoader, Dataset


from threading import Thread
import sys 

from queue import Queue

from imutils.video import FileVideoStream

import numpy as np
import argparse
import imutils
import time
#%% Prepare Data Loader

from data_set_loader import dataPrep

#%% Cuda Determinstic 

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
seed = 42 # any number 
set_deterministic(seed=seed)




#%% Dataset Loader 

with open('../../../../Dataset/ARL_MULTIVIEW_AR/Avijoy_8_26_21/_160_160.pkl', 'rb') as f:
    temp = pickle.load(f)

dataSet =  dataPrep(temp)

data_loader = DataLoader(dataSet, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

#%% Dataloader Check 

sample = next(iter(data_loader))

for i in data_loader:
    break


#%% Image Ploting  (nothing to do with the code)

sample = next(iter(data_loader))

jj =  (sample[0]).cuda()
plt.imshow(np.moveaxis((jj[0,:,0,:,:]).cpu().numpy(), 0,2), vmin=0., vmax=1.)

#%% Random Loss function (add all the loss functions)
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()   




class ContrastiveLoss2(nn.Module):
    None
    
    
    
    
loss = ContrastiveLoss(1)


#%% Network C3D

from C3D_model import C3D
from p3D_net import P3D63, P3D131, P3D199
from torchsummary import  summary

#%% Input Model name

ModName = input("Input your model name C3D, P3D199, P3D63, P3D131 \n")

def str_to_class(ModelName):
    return getattr(sys.modules[__name__], ModelName)

ModelName = str_to_class(ModName)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

if ModName == 'C3D':
    net = ModelName()
    net.load_state_dict(torch.load('../../Saved_models/c3d.pickle'))
    net = net.cuda()
    input_shape = (3,16,112,112)
    print(summary(net, input_shape))

elif ModName == 'P3D199':
    net = ModelName(True, 'RGB',num_classes=400)
    # net = net.cuda()
    # input_shape = (3,16,160,160)
    # print(summary(net, input_shape))

elif ModName == 'P3D63' or ModName == 'P3D131':
    net = ModelName(num_classes=400)
    net.to(device)
    input_shape = (3,16,160,160)
    print(summary(net, input_shape))
    
   # more network here!!
    #SLOWFAST


#%% Removing layer from the mdoel 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.in_features=2048

    def forward(self, x):
        return x
# net.avgpool = Identity() // OR linear!! 
    
net.fc = Identity()

## good link : https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648
## https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6

#%%
num_classes = 100

net.fc = torch.nn.Linear(in_features=2048, out_features= num_classes )

#%% Optimizer 

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

#%% Data inference

# jj =  (i[1].float()/255.0).cuda()



# https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/15

# with torch.no_grad():
#%% 

import torch.nn as nn

z= torch.from_numpy(np.array(1))

L=  torch.nn.TripletMarginLoss(p = 2, margin = 5)

z = z.cuda()

net = net.cuda()
sig_m =  nn.Sigmoid()
j = 0
import pdb
for sample in data_loader:    
    sample = sample.cuda()
    optimizer.zero_grad()   # zero the gradient buffers
    output = sig_m(net(sample[0]))
    loss1 = L(output[0:1], output[1:2],output[2:3])
    loss1.backward()
    if j%500==0:
        print(loss1.cpu())
        # pdb.set_trace()
    optimizer.step()   #%%
    j= j+1







#%% Clear Memory for torch (GPU memory release) (nothing to do with the code)
# del output
# del jj 
# del net

torch.cuda.empty_cache()

# check  devices 
next(net.parameters()).is_cuda
next(net.parameters()).device

#%% Manual test DataLoader (May Not useful)

def get_data_test(idx= None):
    
    # Time Positives
    if idx ==None:
        idx = randint(0,len(temp[0])-100)
        
    x_v1 = temp[0][idx:idx+16]
    x_v2 = temp[1][idx:idx+16]
    x_v3 = temp[2][idx:idx+16]
    
    # Augmentation Positive
    # Horizontal flip 
    x_v1_hf = np.flip(x_v3, axis = 2)
    x_v1_br = brightness_augment(x_v1, factor = 1.5)
    
    x_v1_snp =  snp_RGB(x_v1)
    # Time Negatives 
    idx_n =  randint(0,len(temp[0])-100)
    while abs(idx-idx_n)<1200: idx_n = randint(0,len(temp[0])-100)
    
    xNIra =  temp[0][idx_n:idx_n+16] # intra negative 
    
    # Augmentation Negative 
    # use tensor append option
    # ret = np.moveaxis(np.stack((x_v1, x_v2, x_v3, x_v1_hf, x_v1_br, x_v1_snp, xNIra), axis = 0), -1, -4)
    
    ret = np.moveaxis(np.stack((x_v1, x_v2, x_v3), axis = 0), -1, -4)
    return ret.astype(np.float32)/255.0

#%% Preparing test dataset (change for every new dataset)
# test result
# a = np.int16([0, 1650 , 3240, 4530, 6240,7980,9180,10980, 11580,12030])

a = np.int16([0, 36*30 , 76*30, 115*30, 156*30,201*30,270*30,316*30, 370*30,435*30])


data_t =[]
data_lab = []


for i in range(a.shape[0]-1):
    for _ in range(20):
        sample = torch.from_numpy(get_data_test(randint(a[i], a[i+1])))
        sample = sample.cuda()
        with torch.no_grad():
            output = net(sample)
        data_t.append(output.cpu().numpy())
        data_lab.append([i,i,i])


data_t = np.array(data_t)
data_lab = np.int16(data_lab)
h,w,l = data_t.shape
data_t = data_t.reshape((h*w,l))
data_lab = data_lab.reshape((h*w,1))


#%% TSNE PLOT

from sklearn.manifold import TSNE

label3 = np.int16([ i for i in range(h*w)])%3

def tsne_plot(data = data_t, n_comp = 2, label = label3):
    X_embedded = TSNE(n_components=n_comp, verbose=1).fit_transform(data)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    if n_comp == 3:ax = fig.add_subplot(projection ='3d')
    
    # cdict = {0: 'red', 1: 'blue', 2: 'green'}
    
    markers = ['v', 'x', 'o', '.', '>', '<', '1', '2', '3']
    
    for i, g in enumerate(np.unique(label3)):
        ix = np.where(label == g)
        if n_comp==3:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], X_embedded[ix,2], marker = markers[i], label = g, alpha = 0.8)
        else:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], marker = markers[i], label = g, alpha = 0.8)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    if n_comp==3:ax.set_zlabel('Z Label')
    
    ax.legend(fontsize='large', markerscale=2)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    if n_comp == 3:ax = fig.add_subplot(projection ='3d')
    
    # cdict = {0: 'red', 1: 'blue', 2: 'green'}
    markers = ['v', 'x', 'o', '.', '>', '<', '1', '2', '3']
    
    for i, g in enumerate(np.unique(label)):
        ix = np.where(label == g)
        if n_comp==3:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], X_embedded[ix,2], marker = markers[i], label = g, alpha = 0.8)
        else:
            ax.scatter(X_embedded[ix,0], X_embedded[ix,1], marker = markers[i], label = g, alpha = 0.8)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    if n_comp==3:ax.set_zlabel('Z Label')
    
    
    
    ax.legend(fontsize='large', markerscale=2)
    plt.show()

    

    
    
tsne_plot(1/(1+np.exp(-data_t)), 2, data_lab)
    
    
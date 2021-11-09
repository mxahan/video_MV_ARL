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
#%% Augmentations


# Brighness augmentation video, input shape # frame number, H, W, C

def brightness_augment(img, factor=0.5): 
    rgb = img.copy()
    for i in range(img.shape[0]):
        hsv = cv2.cvtColor(img[i], cv2.COLOR_RGB2HSV) #convert to hsv
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * (factor) #scale channel V uniformly
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
        rgb[i] = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb




# Salt and Papper Noise 
import PIL

class SaltAndPepperNoise(object):
    r""" Implements 'Salt-and-Pepper' noise
    Adding grain (salt and pepper) noise
    (https://en.wikipedia.org/wiki/Salt-and-pepper_noise)

    assumption: high values = white, low values = black
    
    Inputs:
            - threshold (float):
            - imgType (str): {"cv2","PIL"}
            - lowerValue (int): value for "pepper"
            - upperValue (int): value for "salt"
            - noiseType (str): {"SnP", "RGB"}
    Output:
            - image ({np.ndarray, PIL.Image}): image with 
                                               noise added
    """
    def __init__(self,
                 treshold:float = 0.005,
                 imgType:str = "cv2",
                 lowerValue:int = 5,
                 upperValue:int = 250,
                 noiseType:str = "SnP"):
        self.treshold = treshold
        self.imgType = imgType
        self.lowerValue = lowerValue # 255 would be too high
        self.upperValue = upperValue # 0 would be too low
        if (noiseType != "RGB") and (noiseType != "SnP"):
            raise Exception("'noiseType' not of value {'SnP', 'RGB'}")
        else:
            self.noiseType = noiseType
        super(SaltAndPepperNoise).__init__()

    def __call__(self, img1):
        img = img1.copy()
        if self.imgType == "PIL":
            img = np.array(img)
        if type(img) != np.ndarray:
            raise TypeError("Image is not of type 'np.ndarray'!")
        
        if self.noiseType == "SnP":
            random_matrix = np.random.rand(img.shape[0],img.shape[1])
            img[random_matrix>=(1-self.treshold)] = self.upperValue
            img[random_matrix<=self.treshold] = self.lowerValue
        elif self.noiseType == "RGB":
            random_matrix = np.random.random(img.shape)      
            img[random_matrix>=(1-self.treshold)] = self.upperValue
            img[random_matrix<=self.treshold] = self.lowerValue
        
        

        if self.imgType == "cv2":
            return img
        elif self.imgType == "PIL":
            # return as PIL image for torchvision transforms compliance
            return PIL.Image.fromarray(img)

# Define the SNP noises for video 

def snp_RGB(vid_fr):
    vid_sn = vid_fr.copy()
    RGB_noise = SaltAndPepperNoise(noiseType="RGB")
    for i in range(vid_fr.shape[0]):
        vid_sn[i] = RGB_noise(vid_fr[i])
    return vid_sn 
    
#%% Pytorch DataLoader [very Crutial]


class dataPrep(Dataset):
    def __init__(self, root_dir):
        self.all_cam = root_dir
        self.mini =  min(len(root_dir[0]), len(root_dir[2]), len(root_dir[1]))
        
    def __len__(self):
        return len(self.all_cam[0])
        
    def __getitem__(self,idx):
        # x = self.all_cam[randint(0,2)][idx:idx+16]
        # x = self.all_cam[0][idx:idx+16]
        # y = self.all_cam[1][idx:idx+16]
        # z = self.all_cam[2][idx:idx+16]
        xx = self.get_data1()
        return xx
    
    def get_data1(self):
        
        # Time Positives

        idx = randint(0,self.mini-100)

        
        
        # x_v1 = self.all_cam[0][idx:idx+16]
        # x_v2 = self.all_cam[1][idx:idx+16]
        # x_v3 = self.all_cam[2][idx:idx+16]
        
        aps = random.sample(set([0,1,2]),2)
        
        Anchor = self.all_cam[aps[0]][idx:idx+16]
        
        if np.random.uniform()>(0.2-np.exp(-10)):
            Pos = self.all_cam[aps[1]][idx:idx+16]
            
        else:
            ps = randint(0,2)
            if ps ==1:
                Pos = np.flip(Anchor, axis = 2)
            elif ps == 2:
                Pos = brightness_augment(Anchor, factor = 1.5)
            else:
                Pos = snp_RGB(Anchor)
                
        
        ###  Augmentation Positive
        

        # x_v1_hf = np.flip(x_v3, axis = 2)
        # x_v1_br = brightness_augment(x_v1, factor = 1.5)
        # x_v1_snp =  snp_RGB(x_v1)
        
        
        # Time Negatives 
        idx_n =  randint(0,self.mini-100)
        while abs(idx-idx_n)<1200: idx_n = randint(0,self.mini-100)
        
        # xNIra1 =  self.all_cam[0][idx_n:idx_n+16] # intra negative 
        # xNIra2 =  self.all_cam[1][idx_n:idx_n+16] 
        # xNIra3 =  self.all_cam[2][idx_n:idx_n+16] 
        
        ns = randint(0,2)
        
        Neg = self.all_cam[ns][idx_n:idx_n+16]
        
        
        
        # Augmentation Negative 
        
        # use tensor append option
        
        # ret = np.moveaxis(np.stack((x_v1, x_v2, x_v3, x_v1_hf, x_v1_br, x_v1_snp, xNIra), axis = 0), -1, -4)
        
        # ret = np.moveaxis(np.stack((x_v1, x_v2, x_v3), axis = 0), -1, -4)
        
        ret = np.moveaxis(np.stack((Anchor, Pos, Neg), axis = 0), -1, -4)
        
        
        return ret.astype(np.float32)/255.
    
# change the data shape using torch.moveaxis, numpy.moveaxis
# https://pytorch.org/docs/stable/generated/torch.moveaxis.html
# https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html




# sampling for SIMCLR

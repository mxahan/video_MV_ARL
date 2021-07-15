# Import libraries 

#%% libraries

import tensorflow as tf

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

import numpy as np

import cv2

import glob

from scipy.io import loadmat

import random

from random import seed, randint

from sklearn.model_selection import train_test_split

import pandas as pd


#%% Directory load data

files = []

path_dir = '../../../../Dataset/ARL_MULTIVIEW_AR/Trail1_6_19/'

dataPath = os.path.join(path_dir, '*.mp4')
files = glob.glob(dataPath)  # care about the serialization


dataPath = os.path.join(path_dir, '*.MP4')

files.extend(glob.glob(dataPath))  # care about the serialization

#%% Import data

data = []
im_size = (100,100)

cap = cv2.VideoCapture(files[1])

import pdb

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
   # gray  = gray[:,:,:]
    gray =  gray[:,:,:]
   
    gray = cv2.resize(gray, im_size)
    

   
    data.append(gray)
    
    # pdb.set_trace()
    
    cv2.imshow('frame', gray)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


fps = cap.get(cv2.CAP_PROP_FPS)
    
cap.release()
cv2.destroyAllWindows()
data =  np.array(data)

#%% Data Analysis

data.nbytes
data.shape 
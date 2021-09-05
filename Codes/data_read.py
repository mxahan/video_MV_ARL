# Import libraries 

#%% libraries

#import tensorflow as tf

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

#%% Directory load data

files = []

path_dir = '../../../../Dataset/ARL_MULTIVIEW_AR/Trial2_7_27/'

dataPath = os.path.join(path_dir, '*.mp4')
files = glob.glob(dataPath)  # care about the serialization


dataPath = os.path.join(path_dir, '*.MP4')

files.extend(glob.glob(dataPath))  # care about the serialization

#%% Import data
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# https://data-flair.training/blogs/python-project-real-time-human-detection-counting/

data = []
im_size = (400,400)

cap = cv2.VideoCapture(files[2])

import pdb

time.sleep(1.0)
fps = FPS().start()

while(cap.isOpened()):
    ret, gray = cap.read()
    
    if ret==False:
        break

    # pdb.set_trace()
    gray  = gray[:,:,:]
    # gray[150:900,410:1200,:]     
   
    gray = cv2.resize(gray, im_size) 
    
    # pdb.set_trace()
    
    # gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    
    boxes, weights = hog.detectMultiScale(gray, winStride=(4,4), scale = 1.03 ) #may add padding
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(gray, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    
    data.append(gray)
    
    # pdb.set_trace([]

    cv2.imshow('frame', gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup


fps = cap.get(cv2.CAP_PROP_FPS)
    
cap.release()
cv2.destroyAllWindows()
data =  np.array(data)

#%% Data Analysis

print(data.nbytes)
print(data.shape) 

dr_data = data[382:]


#sp_data
#ac_data
#dr_data


# data_pre =  [sp_data, ac_data, dr_data] # Keep the order as it is


#%% Data Save Option

# https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/

## Quick save an load "PICKLE"
input("pickle save ahead")
# Saving 


with open('../../../../Dataset/ARL_MULTIVIEW_AR/Trial2_7_27.pkl', 'wb') as f:
    pickle.dump([sp_data, ac_data, dr_data], f)  #NickName: SAD


# Loading 

with open('../../../../Dataset/ARL_MULTIVIEW_AR/Trial2_7_27.pkl', 'rb') as f:
    sp, ac, dr = pickle.load(f)
    
    
## Data Frame Option (not good option as CSV turns things to string)

# dict = {'sp': [sp_data], 'ac': [ac_data], 'dr': [dr_data]}
# df = pd.DataFrame(dict)

# df.to_csv('file_name.csv')

# df1 =  pd.read_csv('file_name.csv')

# sp = np.array(df1['sp'])


#device = torch.device('cuda:0')

#%% Prepare Data Loader

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

#%% Faster loading (actually not so good in practice!)
'''


data1 = []
fvs = FileVideoStream(files[1],queue_size = 128*32).start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()


# loop over frames from the video file stream
while fvs.more():
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale (while still retaining 3
    # channels)
    frame = fvs.read()
    if not fvs.more():
        break
        
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, im_size)
    # frame = np.dstack([frame, frame, frame])
    data1.append(frame)
    # display the size of the queue on the frame
    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    fps.update()

    
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
data1 =  np.array(data1)


'''

#%% Pytorch DataLoader [very Crutial]



class dataPrep(Dataset):
    def __init__(self, root_dir):
        self.all_cam = root_dir
        
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
        idx = randint(0,len(temp[0])-100)
        x_v1 = torch.from_numpy(temp[0][idx:idx+16])
        x_v2 = torch.from_numpy(temp[1][idx:idx+16])
        x_v3 = torch.from_numpy(temp[2][idx:idx+16])
        
        # Augmentation Positive 
        
        x_v1_hf = torch.from_numpy(np.flip(x_v1, axis = 2))
        
        
        # Time Negatives 
        idx_n =  randint(0,len(temp[0])-100)
        while abs(idx-idx_n)>1200: idx_n = randint(0,len(temp[0])-100)
        
        xNIra =  torch.from_numpy(temp[0][idx_n:idx_n+16]) # intra negative 
        
        # Augmentation Negative 
        
        # use tensor append option
        
        return torch.stack((x_v1, x_v2, x_v3, x_v1_hf, xNIra))

# sampling for SIMCLR


with open('../../../../Dataset/ARL_MULTIVIEW_AR/Trial2_7_27.pkl', 'rb') as f:
    temp = pickle.load(f)

dataSet =  dataPrep(temp)

data_loader = DataLoader(dataSet, batch_size=4, shuffle=True, num_workers=4)


#%% Dataloader Check 

for i in data_loader:
    break

#%% Manual DataLoader


def get_data():
    idx = randint(0,len(temp[0])-100)
    x_v1 = torch.from_numpy(temp[0][idx:idx+16])
    x_v2 = torch.from_numpy(temp[1][idx:idx+16])
    x_v3 = torch.from_numpy(temp[2][idx:idx+16])
    return torch.stack((x_v1, x_v2, x_v3))
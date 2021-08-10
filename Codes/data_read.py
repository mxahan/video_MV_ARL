# Import libraries 

#%% libraries

import tensorflow as tf

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ['KMP_DUPLICATE_LIB_OK']='True'
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

data = []
im_size = (256,256)

cap = cv2.VideoCapture(files[0])

import pdb

time.sleep(1.0)
fps = FPS().start()

while(cap.isOpened()):
    ret, gray = cap.read()
    
    if ret==False:
        break

    
   # gray  = gray[:,:,:]
    gray =  gray[10:1050,600:1800,:]   
    
   
    gray = cv2.resize(gray, im_size) 
    
    # pdb.set_trace()
    
    # gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    
    # boxes, weights = hog.detectMultiScale(gray, winStride=(15,15), scale = 1.03 )
    # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    # for (xA, yA, xB, yB) in boxes:
    #     cv2.rectangle(gray, (xA, yA), (xB, yB),
    #                       (0, 255, 0), 2)
    
    
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

data.nbytes
data.shape 

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

#%%
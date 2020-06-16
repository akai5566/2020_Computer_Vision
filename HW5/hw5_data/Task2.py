#!/usr/bin/env python
# coding: utf-8

# # Bag of SIFT representation + nearest neighbor classifier

# In[2]:


import argparse as ap
import cv2
import imutils 
import numpy as np
import os
import glob
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from __future__ import print_function
import matplotlib.pyplot as plt
import time
from sklearn.externals import joblib


# ## Detect train data feature

# In[ ]:


des_list = []
path = "hw5_data/train/**/*"

t1 = time.time()
files = glob.glob(path)
sift = cv2.xfeatures2d.SIFT_create()
for File in files:
    im = cv2.imread(File)
    #im = cv2.resize(im, (200,200), interpolation = cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(im,None)
    des_list.append((File, des1))
    #print(len(des1))

descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    if descriptor is None:
        print(0)
        continue
    #print(descriptor.shape)
    descriptors = np.vstack((descriptors, descriptor))
    #print(descriptor.shape)
#print(descriptors)
t2 = time.time()
print("time:", t2-t1)


# In[ ]:


print(descriptors.shape)


# ## Detect test data feature

# In[ ]:


test_list = []
path = "hw5_data/test/**/*"

t1 = time.time()
files = glob.glob(path)
sift = cv2.xfeatures2d.SIFT_create()
for File in files:
    im = cv2.imread(File)
    #im = cv2.resize(im, (200,200), interpolation = cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(im,None)
    test_list.append((File, des1))
    #print(des1)
    
t2 = time.time()
print("time:", t2-t1)


# ## K means for all feature

# In[ ]:


# Perform k-means clustering
t1 = time.time()
k = 300
voc, variance = kmeans(descriptors, k, 1) 
t2 = time.time()
print("time:", t2-t1)


# ## Enlarge feature number

# In[ ]:


des_list = []
path = "hw5_data/train/**/*"

t1 = time.time()
files = glob.glob(path)
sift = cv2.xfeatures2d.SIFT_create()
for File in files:
    im = cv2.imread(File)
    im = cv2.resize(im, (600,600), interpolation = cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(im,None)
    des_list.append((File, des1))
    #print(len(des1))

descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    if descriptor is None:
        print(0)
        continue
    #print(descriptor.shape)
    descriptors = np.vstack((descriptors, descriptor))
    #print(descriptor.shape)
#print(descriptors)
t2 = time.time()
print("time:", t2-t1)


# In[ ]:


test_list = []
path = "hw5_data/test/**/*"

t1 = time.time()
files = glob.glob(path)
sift = cv2.xfeatures2d.SIFT_create()
for File in files:
    im = cv2.imread(File)
    im = cv2.resize(im, (1000,1000), interpolation = cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(im,None)
    test_list.append((File, des1))
    #print(des1)
    
t2 = time.time()
print("time:", t2-t1)


# ## Load

# In[332]:


#voc, var = joblib.load('k300.pkl')
#k = 300


# In[244]:


#des_list = joblib.load('train_list600.pkl')
#k = 300


# In[252]:


#test_list = joblib.load('test_list1000.pkl')
#k = 300


# ## Histogram of features based on K means center for each training image

# In[333]:


# Calculate the histogram of features
im_features = np.zeros((1500, k), "float32")
for i in range(1500):
    if des_list[i][1] is None:
        continue
    words, distance = vq(des_list[i][1],voc)

    for w in words:
        im_features[i][w] += 1
    #im_features[i] /= np.sum(im_features[i])
    #im_features[i] /= np.sqrt(np.sum(im_features[i]**2))
    im_features[i] = (im_features[i] - np.mean(im_features[i])) / np.std(im_features[i])
    
print(im_features)


# ## Histogram of features based on K means center for each testing image

# In[334]:


# Calculate the histogram of features
test_features = np.zeros((150, k), "float32")
for i in range(150):
    if test_list[i][1] is None:
        continue
    words, distance = vq(test_list[i][1],voc)
    
    for w in words:
        test_features[i][w] += 1
    #test_features[i] /= np.sum(test_features[i])
    #test_features[i] /= np.sqrt(np.sum(test_features[i]**2))
    test_features[i] = (test_features[i] - np.mean(test_features[i])) / np.std(test_features[i])

print(test_features)


# ## KNN classifier

# In[335]:


def Euclidian(a, b):
    return np.sqrt(np.sum((a-b)**2))
    #return np.linalg.norm(a-b)

def KNN(test, center, K):
    dtype = [('dis', float), ('idx', int)]
    distance = np.array([(Euclidian(test, center[i]),  i) for i in range(len(center))], dtype=dtype)
    #print (distance)
    newdistance = np.sort(distance, order='dis')
    #print (newdistance)
    
    class_count = np.zeros(15)
    for i in range(K):
        _, idx = newdistance[i]
        class_count[idx//100] += 1
        
    #print (class_count)
    #print (np.argmax(class_count))
    return np.argmax(class_count)
    
minIdx = 0
count = 0.
total = 0.

for k in range(0,100,5):
    total = 0.
    for i in range(15):
        count = 0.
        for j in range(10):       
            minIdx = KNN(test_features[i*10+j], im_features, k)
            #print(minIdx)
            
            if minIdx == i:
                count += 1.
        #print(i,count/10.)
        total += count
    print(k, "total:", total/150.)


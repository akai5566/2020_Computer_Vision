from __future__ import print_function
import argparse as ap
import cv2
import imutils 
import numpy as np
import os
import glob
from scipy.cluster.vq import *
import matplotlib.pyplot as plt
import time
import seaborn as sns; sns.set()

label_type = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 'InsideCity', 'Kitchen', 'LivingRoom', 'Mountain', 'Office','OpenCountry', 'Store', 'Street', 'Suburb', 'TallBuilding']
save_dir = "result/Task2"


#Detect train data feature
des_list = []
path = "train/**/*"

t1 = time.time()
files = glob.glob(path)
sift = cv2.xfeatures2d.SIFT_create()
for File in files:
    im = cv2.imread(File)
    im = cv2.resize(im, (600,600), interpolation = cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(im,None)
    des_list.append((File, des1))
    #print(len(des_list))
    

des_list_0 = des_list[0]
descriptors = des_list_0[1]

for image_path, descriptor in des_list[1:]:
    if descriptor is None:
        print(0)
        continue
    #print(descriptor.shape)
    descriptors = np.vstack((descriptors, descriptor))
    #print(descriptor.shape)
    #print(descriptors)

t2 = time.time()
print("Detect train data time:", t2-t1)

#print(descriptors.shape)


#Detect test data feature
test_list = []
path = "test/**/*"

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
print("Detect test data time:", t2-t1)

# K means for all feature
# Perform k-means clustering
t1 = time.time()
k = 295
voc, variance = kmeans(descriptors, k, 1) 
t2 = time.time()
print("Perform kmeans clustering time:", t2-t1)


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
    
#print(im_features)

# Histogram of features based on K means center for each testing image
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

#print(test_features)


# KNN classifier

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

    plot_heatmap(midTdx, total, save_dir )
    plot_res(midTdx, total, save_dir)



def plot_heatmap(true_y, pred_y, save_dir):
    sns.heatmap(confusion_matrix(true_y, pred_y, labels=label_type, normalize='true'),xticklabels=label_type,yticklabels=label_type)
    plt.tight_layout()
    plt.savefig(save_dir)

def plot_res(true_y, pred_y, save_dir='res'):

    def unique_by_key(elements, key=None):
        if key is None:
            # no key: the whole element must be unique
            key = lambda e: e
        return list({key(el): el for el in elements}.values())

    train = []
    test = []

    train_dict = {}
    test_dict = {}

    for index, label in enumerate(label_type):
        training_imgs = glob.glob('train/{}/*.jpg'.format(label))
        testing_imgs = glob.glob('test/{}/*.jpg'.format(label))
        for fname in training_imgs:
            img = cv2.imread(fname)
            train.append(img)
            if label not in train_dict:
                train_dict[label] = img

        for fname in testing_imgs:
            img = cv2.imread(fname)
            test.append(img)
            if label not in test_dict:
                test_dict[label] = img

    false_negative = {k:[] for k in label_type}
    false_positive = {k:[] for k in label_type}
    true_positive = {k:[] for k in label_type}

    for idx in range(len(true_y)):
        if true_y[idx] != pred_y[idx]:
            false_negative[true_y[idx]].append((idx,pred_y[idx]))
            false_positive[pred_y[idx]].append((idx,true_y[idx]))
        else:
            true_positive[true_y[idx]].append(idx)

    for cat in false_negative:
        false_negative[cat]=unique_by_key(false_negative[cat], key=itemgetter(1))

    for cat in false_positive:
        false_negative[cat]=unique_by_key(false_negative[cat], key=itemgetter(1))

    fig, axes = plt.subplots(nrows=16, ncols=5, figsize=(12, 30))

    axes[0][0].axis('off')

    for idx, cat in enumerate(label_type):

        axes[idx+1][1].axis('off')
        axes[idx+1][1].imshow(train_dict[cat])

        axes[idx+1][2].axis('off')
        if len(true_positive[cat])!=0:
            axes[idx+1][2].imshow(test[true_positive[cat][0]])

        axes[idx+1][3].axis('off')
        if len(false_positive[cat])!=0:
            axes[idx+1][3].set_title(false_positive[cat][0][1])
            axes[idx+1][3].imshow(test[false_positive[cat][0][0]])

        axes[idx+1][4].axis('off')
        axes[idx+1][4].patch.set_facecolor('xkcd:mint green')
        if len(false_negative[cat])!=0:
            axes[idx+1][4].set_title(false_negative[cat][0][1])
            axes[idx+1][4].imshow(test[false_negative[cat][0][0]])

    for ax, row in zip(axes[1:,0], label_type):
        ax.axis('off')
        ax.set_title(row, rotation=0, size='large',fontweight='bold',loc='right')

    for ax, col in zip(axes[0][1:], ["Sample training images","Sample true positives","False positives with \ntrue label",'False negatives with \nwrong predicted label']):
        ax.axis('off')
        ax.set_title(col, rotation=0, size='large',fontweight='bold',y=-0.01)

    fig.tight_layout()
    plt.savefig(save_dir)
    plt.show()


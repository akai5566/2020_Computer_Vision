#!/usr/bin/env python
# coding: utf-8

# In[5]:

from __future__ import print_function
#get_ipython().run_line_magic('matplotlib', 'inline')



import vis
import math
import keras
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import keras
from keras.applications import VGG16
from keras import optimizers
from keras import activations
from keras import backend as K
from keras.models import Sequential
from keras.utils import to_categorical
from vis.visualization import visualize_activation
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D


# In[ ]:


#=====Load training data=========

train_data = []
path = "train/**/*"

#====Get the file name which under the folder===
files = glob.glob(path)

for File in files:
    im = Image.open(File)
    
######Resize the images to a common size######
    im = im.resize((224,224),Image.BICUBIC) # 224 initial
    im = np.asarray(im).astype('float32')
    
#=======Normalize the value of images========
    im = im/255.
    
#=== Transpose the images to channel first, it will be fast at training ===
    train_data.append(im)#np.transpose(im, (2, 0, 1)))
    
train_data = np.asarray(train_data)
print("Training data size:",train_data.shape)


# In[ ]:


#=====Load testing data=========

test_data = []
path = "test/**/*"

#====Get the file name which under the folder===
files = glob.glob(path)

for File in files:
    im = Image.open(File)
    
######Resize the images to a common size######
    im = im.resize((224,224),Image.BICUBIC)
    im = np.asarray(im).astype('float32')
    
#=======Normalize the value of images========
    im = im/255.
    
#=== Transpose the images to channel first, it will be fast at training ===
    test_data.append(im)#np.transpose(im, (2, 0, 1)))
    
test_data = np.asarray(test_data)
print("Testing data size:",test_data.shape)


# In[5]:


train_label = np.zeros((1500))
for i in range(15):
    train_label[i*100:100*i+100] = i
print("Train label size:",train_label.shape)
print(train_label)


# In[6]:


test_label = np.zeros((150))
for i in range(15):
    test_label[i*10:10*i+10] = i
print("Test label size:",test_label.shape)
print(test_label)


# In[7]:


batch_size = 128
num_classes = 15
epochs = 100

img_rows, img_cols = 224, 224
img_shape = (img_rows, img_cols)
input_shape = (img_rows, img_cols, 1)


# In[8]:


train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
print("Training data size:",train_data.shape)
print("Testing data size:",test_data.shape)


# In[9]:


train_label = keras.utils.to_categorical(train_label, num_classes)
test_label = keras.utils.to_categorical(test_label, num_classes)
print(train_label)
print(test_label)


# ### Build our Model

# In[63]:


'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

#from _future_ import print_function
#import keras
#from keras.datasets import cifar10
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#import os

batch_size = 32
num_classes = 15
epochs = 300
data_augmentation = True
num_predictions = 20
#save_dir = os.path.join(os.getcwd(), 'saved_models')
#model_name = 'keras_cifar10_trained_model.h5'
#
## The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')
#
## Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=train_data.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(train_data)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(train_data, train_label,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(test_data, test_label),
                        workers=4)

# Save model and weights
#if not os.path.isdir(save_dir):
#    os.makedirs(save_dir)
#model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
#print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(test_data, test_label, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[64]:


K.set_value(model.optimizer.lr, 0.00001)
model.fit_generator(datagen.flow(train_data, train_label,
                                     batch_size=batch_size),
                        epochs=50,
                        validation_data=(test_data, test_label),
                        workers=4)


# In[65]:


K.set_value(model.optimizer.lr, 0.000001)
model.fit_generator(datagen.flow(train_data, train_label,
                                     batch_size=batch_size),
                        epochs=50,
                        validation_data=(test_data, test_label),
                        workers=4)


# In[66]:


# Testing
score = model.evaluate(test_data, test_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


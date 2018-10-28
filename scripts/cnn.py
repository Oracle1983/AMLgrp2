import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting

x_train = []
y_train = []
final_img_count_train = 0
seq_img_count  = 0
ges_img_count = 0
sub_img_count = 0
img_count = 0
for i in range(0, 2): # select 2 subject areas only
    ges_img_count = 0
    for j in os.listdir('./0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            seq_img_count = 0 # To tally images of a given gesture
            for k in os.listdir('./0' +
                                str(i) + '/' + j + '/' + 'train' + '/'):
                if not k.startswith('.'): # Again avoid hidden folders
                    img_count = 0
                    for l in os.listdir('./0' +
                                    str(i) + '/' + j + '/' + 'train' + '/' + k + '/'):
                        img = Image.open('./0' +
                                     str(i) + '/' + j + '/' + 'train' + '/' + k + '/' + l).convert('L')
                        # Read in and convert to greyscale
                        img = img.resize((320, 120))
                        arr = np.array(img)
                        x_train.append(arr)
                        img_count = img_count + 1
                        #y_values = np.full((img_count, 1), lookup_seq[j])
                        #y_values = lookup_seq[j]
                        y_values = int(j)
                        y_train.append(y_values)
                    seq_img_count = seq_img_count + img_count
            ges_img_count = ges_img_count + seq_img_count
            #print("ges_img_count",ges_img_count)
    sub_img_count = sub_img_count + ges_img_count
    #print("sub_img_count",sub_img_count)
final_img_count_train = final_img_count_train + sub_img_count
print(final_img_count_train)
x_train = np.array(x_train, dtype = 'float32')
y_train = np.array(y_train)
#y_train = y_train.reshape(final_img_count_train, 1) # Reshape to be the correct size
#y_train = y_train.reshape(1, -1)

import keras
from keras.utils import to_categorical
y_train = to_categorical(y_train)
print(x_train.shape)
print(y_train.shape)
x_train.shape[1:]

x_test = []
y_test = []
final_img_count_test = 0
seq_img_count  = 0
ges_img_count = 0
sub_img_count = 0
img_count = 0
for i in range(0, 2): # select 2 subject areas only
    ges_img_count = 0
    for j in os.listdir('./0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            seq_img_count = 0 # To tally images of a given gesture
            for k in os.listdir('./0' +
                                str(i) + '/' + j + '/' + 'test' + '/'):
                if not k.startswith('.'): # Again avoid hidden folders
                    img_count = 0
                    for l in os.listdir('./0' +
                                    str(i) + '/' + j + '/' + 'test' + '/' + k + '/'):
                        img = Image.open('./0' +
                                     str(i) + '/' + j + '/' + 'test' + '/' + k + '/' + l).convert('L')
                        # Read in and convert to greyscale
                        img = img.resize((320, 120))
                        arr = np.array(img)
                        x_test.append(arr)
                        img_count = img_count + 1
                        #y_values = np.full((img_count, 1), lookup_seq[j])
                        #y_values = lookup_seq[j]
                        y_values = int(j)
                        y_test.append(y_values)
                    seq_img_count = seq_img_count + img_count
            ges_img_count = ges_img_count + seq_img_count
            #print("ges_img_count",ges_img_count)
    sub_img_count = sub_img_count + ges_img_count
    #print("sub_img_count",sub_img_count)
final_img_count_test = final_img_count_test + sub_img_count
print(final_img_count_test)
x_test = np.array(x_test, dtype = 'float32')
y_test = np.array(y_test)
#y_test = y_test.reshape(final_img_count_test, 1) # Reshape to be the correct size
#y_test = y_test.reshape(1, -1)

import keras
from keras.utils import to_categorical
y_test = to_categorical(y_test)

x_train = x_train.reshape((final_img_count_train, 120, 320, 1))
x_train /= 255
x_test = x_test.reshape((final_img_count_test,120,320,1))
x_test /= 255
x_train.shape

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(64, kernel_size=(4,4), strides=(1,1) , activation='relu',input_shape = (120,320,1)))
model.add(MaxPooling2D (pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1) , activation='relu'))
model.add(Flatten())
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='softmax'))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = 1
batch_size = 64
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

[loss, acc] = model.evaluate(x_test_1,y_test_1,verbose=1)
print("Accuracy:" + str(acc) + ":" + "Loss:" + str(loss))

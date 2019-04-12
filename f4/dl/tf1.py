# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:21:52 2018 by Dhiraj Upadhyaya for DS 
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
#%%
from keras.datasets import mnist

#set "KERAS_BACKEND=tensorflow"  give this at anaconda prompt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#%%
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
num_pixels = X_train.shape[1] * X_train.shape[2]
num_pixels
#%%
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_train = X_train/255
X_test = X_test/255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# data pre-processed now

#%%
#write code for 2 layer NN
from keras.layers import Dense 
model = Sequential()
# now add layers
model.add(Dense(30, input_dim = num_pixels, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
#%%
model.summary()
model.fit(X_train, y_train, batch_size=200, epochs=10)
scores = model.evaluate(X_test, y_test)
scores

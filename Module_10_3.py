# -*- coding: utf-8 -*-
'''
Retraining (Finetuning) Example with vgg.tflearn. Using weights from VGG model to retrain
network for a new task (your own dataset).All weights are restored except
last layer (softmax) that will be retrained to match the new task (finetuning).

Using pretrained model for further training with other inputs.
There are several approaches to fine tuning - this is one of them.
'''
from __future__ import division, print_function, absolute_import
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
#from tflearn.data_utils import shuffle, to_categorical
from sklearn import datasets
from sklearn.model_selection import train_test_split
num_classes = 10 # num of your dataset

iris   = datasets.load_iris()
data   = iris["data"]
target = iris["target"]

# Prepend the column of 1s for bias
all_X = data



# Convert into one-hot vectors
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!
train_X, test_X, train_y, test_y = train_test_split(all_X, all_Y, test_size=0.33)


# Redefinition of convnet_cifar10 network
network = input_data(shape=[None, 4])
network = fully_connected(network, 500, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 200, activation='relu')
network = dropout(network, 0.5)
softmax = fully_connected(network, 3, activation='softmax')
regression = regression(softmax, optimizer='adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)  

model = tflearn.DNN(regression, tensorboard_verbose=3)

# Start finetuning
model.fit(train_X, train_y, n_epoch=60, validation_set=(test_X, test_y), shuffle=True,
          show_metric=True, batch_size=1, snapshot_step=200,
          snapshot_epoch=False, run_id='cifar_apply')

model.save('./models/cifar_apply_1')
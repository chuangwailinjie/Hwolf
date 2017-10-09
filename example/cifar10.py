# coding:utf-8

# http://opu8lkq3n.bkt.clouddn.com/17-9-27/74840040.jpg

import pickle
import os
import numpy as np


from hwolf.model.models import Sequential,Model
from hwolf.layers.core import FullyConnected,Flatten,Softmax,Input,Dropout,Activation
from hwolf.layers.convolution import Conv2d
from hwolf.layers.pooling import MaxPoolingLayer,AvgPoolingLayer
from hwolf.base.activations import Relu,LeakyRelu,Elu
from hwolf.base.optimizers import Momentum,Adam,SGD

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")#32*32*3
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  Ytr=int2onehot(Ytr)
  Yte=int2onehot(Yte)
  return Xtr, Ytr, Xte, Yte

def int2onehot(data):
    onehot=np.zeros((len(data),10))
    for i in range(len(data)):
        onehot[i][data[i]]=1
    return onehot

def cifar_cnn(ROOT):
    X, y, X_test, y_test =load_CIFAR10(ROOT)

    model = Sequential()
    model.add(Input(batch_input_shape=(None,32,32,3)))
    model.add(Conv2d((5,5),stride=1,filter_num=32,zero_padding=2,activator='relu'))
    model.add(MaxPoolingLayer((3,3),stride=2))
    model.add(Conv2d((5, 5), stride=1,filter_num=32,zero_padding=2, activator='relu'))
    model.add(AvgPoolingLayer((3, 3), stride=2))
    model.add(Conv2d((5, 5), stride=1,filter_num=64,zero_padding=2, activator='relu'))
    model.add(AvgPoolingLayer((3, 3), stride=2))
    model.add(Flatten())
    model.add(FullyConnected(output_dim=64,activator='relu'))
    #model.add(Flatten())
    model.add(Softmax(10))
    model.compile('mle',optimizer=Adam())
    model.fit(X,y,validation_data=(X_test,y_test),verbose=2)


def cifar_cnn_2(ROOT):
    X, y, X_test, y_test =load_CIFAR10(ROOT)

    model = Sequential()
    model.add(Input(batch_input_shape=(None,32,32,3)))
    model.add(Conv2d((3,3),stride=1,filter_num=32,zero_padding=2,activator='relu'))
    model.add(MaxPoolingLayer((3,3),stride=2))
    model.add(Conv2d((4, 4), stride=1,filter_num=64,zero_padding=2, activator='relu'))
    model.add(AvgPoolingLayer((3, 3), stride=2))
    model.add(Flatten())
    model.add(FullyConnected(output_dim=64,activator='relu'))
    model.add(Softmax(10))
    model.compile('mle',optimizer=Adam())
    model.fit(X,y,validation_data=(X_test,y_test),verbose=2)


if __name__=='__main__':
    cifar_cnn('cifar')

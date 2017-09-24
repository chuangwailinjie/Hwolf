import numpy as np

from hwolf.model.models import Sequential,Model
from hwolf.layers.core import FullyConnected,Flatten,Softmax,Input,Dropout,Activation
from hwolf.layers.convolution import Conv2d
from hwolf.layers.pooling import MaxPoolingLayer,AvgPoolingLayer
from hwolf.base.activations import Relu,LeakyRelu,Elu
from hwolf.base.optimizers import Momentum,Adam,SGD

def mlp_mnist():
    # multiple layer perceptron
    from tensorflow.examples.tutorials.mnist import input_data
    mnist=input_data.read_data_sets('/tmp/data',one_hot=True)

    training_data=np.array([image.flatten() for image in mnist.train.images])
    training_label=mnist.train.labels

    valid_data=np.array([image.flatten() for image in mnist.validation.images])
    valid_label=mnist.validation.labels

    input_dim=training_data.shape[1]
    label_size=training_label.shape[1]

    model = Sequential()
    model.add(Input(input_shape=(input_dim, )))
    model.add(FullyConnected(300, activator='leakyrelu'))
    model.add(Dropout(0.2))
    model.add(Softmax(label_size))
    model.compile('mle', optimizer=SGD(lr=1e-2))
    model.fit(training_data, training_label, validation_data=(valid_data, valid_label))

def cnn_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist=input_data.read_data_sets('/tmp/data',one_hot=True)

    training_data = np.array([image.reshape(28, 28, 1) for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.reshape(28, 28, 1) for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    label_size = training_label.shape[1]

    model = Sequential()
    model.add(Input(batch_input_shape=(None,28,28,1)))
    model.add(Conv2d((3,3),1,activator='leakyrelu'))
    model.add(AvgPoolingLayer((2,2),stride=2))
    model.add(Conv2d((4, 4), 2, activator='leakyrelu'))
    model.add(AvgPoolingLayer((2, 2), stride=2))
    model.add(Flatten())
    model.add(Softmax(label_size))
    model.compile('mle',optimizer=SGD(lr=1e-2))
    model.fit(training_data,training_label,validation_data=(valid_data,valid_label),verbose=2)

def model_mlp_mnist():
    """test MLP with MNIST data and Model

    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    training_data = np.array([image.flatten() for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.flatten() for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    input_dim = training_data.shape[1]
    label_size = training_label.shape[1]

    dense_1 = FullyConnected(300, input_dim=input_dim, activator=None)
    dense_2 = Activation('selu')(dense_1)
    dropout_1 = Dropout(0.2)(dense_2)
    softmax_1 = Softmax(label_size)(dropout_1)
    model = Model(dense_1, softmax_1)
    model.compile('mle', optimizer=Adam())
    model.fit(training_data, training_label, validation_data=(valid_data, valid_label))

if __name__=='__main__':
    #cnn_mnist()
    #mlp_mnist()
    model_mlp_mnist()

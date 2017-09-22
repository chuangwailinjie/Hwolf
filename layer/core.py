import numpy as np

from ..base import activations
from ..base.activations import Sigmoid,Identity,Softmax,Relu
from ..base.initializers import gaussian_initializer,xavier_uniform_initializer
from layer import Layer,Input


class FullyConnected(Layer):
    def __init__(self,output_dim,input_dim=None,activator='sigmoid',
                    initializer=gaussian_initializer):
        #input_dim is the sample's shape, and inpur_shape is the n x input_dim
        #n is the batch_size
        super().__init__()
        self.output_dim=output_dim
        self.input_dim=input_dim
        self.activator=activator
        self.initializer=initializer
        self.__input_shape=None
        self.__output_shape=None
        # the first layer's input_dim is not None
        if self.input_dim id not None:
            self.connection(None)

    @property
    def W(self):
        return self.__W

    @W.setter
    def W(self,W):
        self.__W=W

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self,b):
        self.__b=b

    @property
    def delta_W(self):
        return self.__delta_W

    @delta_W.setter
    def delta_W(self,delta_W):
        self.__delta_W=delta_W

    @property
    def delta_b(self):
        return self.__delta_b

    @delta_b.setter
    def delta_b(self,delta_b):
        self.__delta_b=delta_b

    @property
    def delta(self):
        return self.__delta

    @delta.setter
    def delta(self,delta):
        self.__delta=delta

    @property
    def params(self):
        return [self.W,self.b]

    @property
    def grad(self):
        return [self.delta_W,self.delta_b]

    def call(self,pre_layer=None,*args,**kwargs):
        self.connection(pre_layer)
        return self

    def connection(self,pre_layer):
        self.pre_layer=pre_layer
        if pre_layer is None:#this layer is the first layer
            if self.input_dim is None:
                raise ValueError('the first layer must have not none input_dim')
            self.input_shape=[None,self.input_dim]
            self.output_shape=[None,self.output_dim]
        else:
            pre_layer.next_layer=self
            self.input_dim=pre_layer.output_shape[1]
            self.input_shape=pre_layer.output_shape
            self.output_shape=[pre_layer.output_shape[0],self.output_dim]
        self.W=self.initializer([self.output_dim,self.input_dim])
        self.b=self.initializer([self.output_dim])
        self.delta_b=np.zeros([self.output_dim])
        self.delta_W=np.zeros([self.output_dim,self.input_dim])
        #self.delta represtnt inputs_sample(x)'s delta
        self.delta=np.zeros([self.input_dim])

    def forward(self,inputs,*args,**kwargs):
        inputs=np.asarray(inputs)
        if len(inputs.shape)==1:#transformer 1-D tensor to 2-D tnsor
            inputs=inputs[None,:]
        self.input_shape=inputs.shape
        self.output_shape[0]=inputs.shape[0]#the number of batch size
        self.inputs=inputs
        #self.inputs.shape [n,d] self.W.shape [ouput_dim[1],d]
        self.logit=np.dot(self.inputs,self.W.T)+self.b
        self.output=self.activator.forward(self.logit)
        return self.ouput

    def backward(self.pre_delta,*args,**kwargs):
        if len(pre_delta.shape)==1:
            pre_delta=pre_delta[None,:]
        batch_size=self.inputs.shape[0]
        #current layer's activator delta is the pre_layer's delta*activaor
        act_delta=pre_delta*self.activator.backward(self.logit)
        self.delta_W=np.dot(act_delta.T,self.inputs)
        self.delta_b=np.mean(act_delta,axis=0)
        self.delta=np.dot(act_delta,self.W)
        return self.delta

class Softmax(FullyConnected):
    def _init__(self,output_dim,input_dim=None,initializer=gaussian_initializer):
        super().__init__(output_dim==output_dim,input_dim=input_dim,
        activator='softmax',initializer=initializer)


class Flatten(Layer):
    def __init__(self,output_dim,input_dim=None,activator='sigmoid',
                    initializer=gaussian_initializer):
        super().__init__()

    @property
    def params(self):
        return list()

    @property
    def grad(self):
        return list()

    def call(self,pre_layer=None,*args,**kwargs):
        self.connection(pre_layer)
        return self

    def _compute_output_shape(self,input_shape,*args,**kwargs):
        if not all(input_shape[1:]):
            raise ValueError('input shape`s value format error')
        return (input_shape[0],np.prod(input_shape[1:]))

    def connection(self,pre_layer):
        self.pre_layer=pre_layer
        if pre_layer is None:#this layer is the first layer
            raise ValueError('flatten layer can`t be the first layer, must have pre_layer')
        self.pre_layer.next_layer=self
        self.input_shape=self.pre_layer.output_shape
        self.output_shape=_compute_output_shape(self.input_shape)

    def forward(self,inputs,*args,**kwargs):
        self.input_shape=inputs.shape
        self.output_shape=self._compute_output_shape(self.input_shape)
        return np.reshape(inputs,self.output_shape)

    def backward(self.pre_delta,*args,**kwargs):
        return np.reshape(pre_delta,self.input_shape)


class Dropout(Layer):
    def __init__(self,dropout=0.5,axis=None):
        super().__Init__()
        self.dropout=dropout
        self.axis=axis
        self.mask=None

    @property
    def params(self):
        return list()

    @property
    def grad(self):
        return list()

    def call(self,pre_layer=None,*args,**kwargs):
        self.connection(pre_layer)
        return self

    def connection(self,pre_layer):
        if pre_layer is None:
            raise ValueError('.......')
        if self.axis is None:
            self.axis=range(len(pre_layer.output_shape))

        self.output_shape=pre_layer.output_shape
        self.pre_layer.next_layer=self
        self.pre_layer=pre_layer

    def forward(self,inputs,is_train=True,*args,*kwargs):
        self.input=input_shape
        if 0.<self.dropout<1:
            if is_train:
                self.mask=np.random.binomial(1,1-self.dropout,
                np.asarray(self.input.shape)[self.axis])
                #element value need scalar
                return self.mask*self.input/(1-self.dropout)
            else:
                #if is val or test, need scalar
                return self.input*(1-self.dropout)
        else:
            return self.input

    def backward(self,pre_delta,*args,**kwargs):
        if 0.<self.dropout<1.:
            return self.mask*pre_delta
        else:
            return pre_delta


class Activation(Layer):

    def __init__(self,activaor,inputs_sample=None):
        super().__init__()
        self.activaor=activations.get(activaor)
        self.input_shape=input_shape

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def call(self,pre_layer=None,*args,.**kwargs):
        self.connection(pre_layer)
        return self

    def connection(self,pre_layer):
        self.pre_layer=pre_layer
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('input_shape muse needed')

        else:
            pre_layer.next_layer=self
            self.input_shape=pre_layer.output_shape
        self.output_shape=self.input_shape

    def forward(self.inputs,*args,**kwargs):
        inputs=np.asarray(inputs)
        if len(inputs.shape)==1:
            inputs=inputs[None,:]
        assert list(self.input_shape[1:])==list(inputs.shape[1:])
        self.input_shape=inputs.shape
        self.output_shape=self.input_shape
        self.inputs=inputs
        self.output=self.activator.forward(self.inputs)
        return sele.output

    def backward(self,pre_delta,*args,**kwargs):
        if len(inputs.shape)==1:
            inputs=inputs[None,:]
        act_delta=pre_delta*self.activaor.backward(self.inputs)
        return act_delta

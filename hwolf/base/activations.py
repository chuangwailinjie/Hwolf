import numpy as  np

# base class

class Activator(object):
    def forward(self,x,*args,**kwargs):
        raise NotImplementedError('function `forward` should be implemented!')
    def backward(self,x,*args,**kwargs):
        raise NotImplementedError('function `backward` should be implemented!')


# sigmoid

def sigmoid(x):
    z=np.asarray(x)
    return 1.0 /(1 + np.exp(-x))

def back_sigmoid(x):
    z=np.asarray(x)
    return sigmoid(x)*(1.0-sigmoid(x))

class Sigmoid(Activator):
    def forward(self,x,*args,**kwargs):
        return sigmoid(x)
    def backward(self,x,*args,**kwargs):
        return back_sigmoid(x)

# relu

def relu(x,max_value=None):
    z=np.array(x)
    x=np.maximum(x,0)
    if max_value is not None:
        x=np.clip(x,0,max_value)
    return x

def back_relu(x,alpha=0.,max_value=None):
    """if max_value is not None:
        return (np.where(x<=max_value,x)>=0).astype(int)
    else:
        return (x>=0).astype(int)"""
    return (x>=0).astype(int)

class Relu(Activator):
    def __init__(self,alpha=0.,max_value=None):
        self.alpha=alpha
        self.max_value=max_value
    def forward(self,x,*args,**kwargs):
        return relu(x,self.alpha,self.max_value)
    def backward(self,x,*args,**kwargs):
        return back_relu(x,self.alpha,self.max_value)


# leaky relu

def leaky_relu(x,alpha=0.3):
    x=np.asarray(x)
    return np.maximum(0,x)+np.minimum(0,x)*alpha

def back_leaky_relu(x,alpha=0.3):
    x=np.asarray(x)
    return np.greater_equal(x,0).astype(int)+np.less(x,0).astype(int)*alpha

class LeakyRelu(Activator):
    def forward(self,x,*args,**kwargs):
        return leaky_relu(x)
    def backward(self,x,*args,**kwargs):
        return back_leaky_relu(x)


# ELU

def elu(x,alpha=1.0):
    x=np.asarray(x)
    return np.maximum(x,0)+alpha*(np.exp(np.minimum(0,x))-1.0)

def back_elu(x,alpha=1.0):
    x=np.asarray(x)
    return np.greater_equal(0,x).astype(int)+alpha*np.exp(np.minimum(x,0))*np.less(x,0).astype(int)

class Elu(Activator):
    def forward(self,x,*args,**kwargs):
        return elu(x)
    def backward(self,x,*agrs,**kwargs):
        return back_elu(x)



# identity  HiwghwayNetWork ResNet

def identity(x):
    x=np.asarray(x)
    return x

def back_identity(x):
    x=np.asarray(x)
    return np.ones(x.shape)# Note: not np.eye

class Identity(Activator):
    def forward(self,x,*args,**kwargs):
        return identity(x)
    def backward(self,x,*args,**kwargs):
        return back_identity(x)

# softmax && logistic  softmax usually be used in output layer

def softmax(x):
    x=np.asarray(x)
    if len(x.shape)>1:
        x-=x.max(axis=1).reshape([x.shape[0],1])
        x=np.exp(x)
        x/=np.sum(x,axis=1).reshape([x.shape[0],1])
        return x
    else:
        x -=np.max(x)
        x=np.exp(x)
        x/=np.sum(x)
        return x

def back_softmax(x):
    x=np.asarray(x)
    return np.ones(x.shape,dtype=x.dtype)# softmax is the top layer,need not update parameters by gradient

class Softmax(Activator):
    def forward(self,x,*args,**kwargs):
        return softmax(x)
    def backward(self,x,*args,**kwargs):
        return back_softmax(x)


# tanh

def tanh(x):
    x=np.asarray(x)
    return np.tanh(x)

def back_tanh(x):
    x=np.asarray(x)
    return 1-np.power(np.tanh(x),2)

class Tanh(Activator):
    def forward(self,x,*args,**kwargs):
        return tanh(x)
    def backward(self,x,*args,**kwargs):
        return back_tanh(x)

# softplus

def softplus(x):
    x=np.asarray(x)
    return np.log(1+np.exp(x))

def back_softplus(x):
    x=np.asarray(x)
    return 1/(1+np.exp(-x))

class Softplus(Activator):
    def forward(self,x,*args,**kwargs):
        return softplus(x)
    def backward(self,x,*args,**kwargs):
        return back_softplus(x)


def selu(z, alpha, scale):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    z = np.asarray(z)
    return scale * elu(z, alpha)

def delta_selu(z, alpha, scale):
    z = np.asarray(z)
    return scale * delta_elu(z, alpha)

class Selu(Activator):
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, z, *args, **kwargs):
        return selu(z, self.alpha, self.scale)

    def backward(self, z, *args, **kwargs):
        return delta_selu(z, self.alpha, self.scale)


# Dropout

# accept str or instance type
def get(activator):
    if activator is None:
        return Identity()
    if isinstance(activator,str):
        activator=activator.lower()
        if activator in('identity'):
            return Identity()
        elif activator in ('sigmoid'):
            return Sigmoid()
        elif activator in ('relu'):
            return Relu()
        elif activator in ('leakyrelu'):
            return LeakyRelu()
        elif activator in ('elu'):
            return Elu()
        elif activator in ('selu',):
            return Selu()
        elif activator in ('softmax'):
            return Softmax()
        elif activator in ('tanh'):
            return Tanh()
        elif activator in ('softplus'):
            return Softplus()
        else:
            raise ValueError('Unknown activator name"{}"'.format(activator))
    elif isinstance(activator,Activator):
        return activator
    else:
        raise ValueError('Unknown activator type "{}"'.format(activator))

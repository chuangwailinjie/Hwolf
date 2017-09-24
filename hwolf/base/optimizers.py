import numpy as np

class Optimizer(object):
    def __init__(self,lr=1e-3,decay=0.,grad_clip=-1,lr_min=0,lr_max=np.inf):
        self.lr=lr
        self.decay=decay
        self.clip=grad_clip
        self.lr_min=lr_min
        self.lr_max=lr_max
        self.iter=0

    def update(self):
        self.iter+=1
        self.lr*=1.0/(1+self.decay*self.iter)
        self.lr=np.clip(self.lr,self.lr_min,self.lr_max)

class SGD(Optimizer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def minimize(self,params,grads):
        for p,g in zip(params,grads):
            p -= self.lr*_grad_clip(g,self.clip)
        super().update()

    def maximum(self,params,grads):
        for p,g in zip(params,grads):
            p +=self.lr*_grad_clip(g,self.clip)
        super().update()


# Use nesterov
class Momentum(Optimizer):
    def __init__(self,momentum=0.9,nesterov=False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.momentum=momentum
        self.nesterov=nesterov
        self.velocity=dict()

    def minimize(self,params,grads):
        for p,g in zip(params,grads):
            v=self.velocity.get(id(p),np.zeros_like(p))
            v=self.momentum*v-self.lr*g
            if self.nesterov:
                p=p+self.momentum*v-self.lr*g
            else:
                p+=v
            self.velocity[id(p)]=v
        super().update()

    def maximum(self,params,grads):
        pass

class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = dict()
        self.v = dict()

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            m = self.m.get(id(p), np.zeros_like(p))
            v = self.v.get(id(p), np.zeros_like(p))
            self.m[id(p)] = self.beta1 * m + (1 - self.beta1) * g
            self.v[id(p)] = self.beta2 * v + (1 - self.beta2) * g ** 2
            mb = self.m[id(p)] / (1 - self.beta1 ** (self.iter + 1))
            vb = self.v[id(p)] / (1 - self.beta2 ** (self.iter + 1))
            p -= (self.lr * mb / (np.sqrt(vb) + self.epsilon))
        super(Adam, self).update()

    def maximum(self,params,grads):
        pass



sgd = SGD
momentum = Momentum
adam = Adam

def get(optimizer):
    if isinstance(optimizer,str):
        optimizer=optimizer.lower()
        if optimizer in ('sgd'):
            return SGD()
        elif optimizer in ('adam'):
            return Adam()
        elif optimizer in ('momentum'):
            return Momentum()
        else:
            raise ValueError('Unknow optimizer name `{}`'.format(optimizer))
    elif isinstance(optimizer,Optimizer):
        return optimizer
    else:
        raise ValueError('Unknown optimizer type`{}`'.format(optimizer.__class__.__name__))




def _grad_clip(grad,clip):
    if clip>0:
        return np.clip(grad,-clip,clip)
    return grad

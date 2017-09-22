import nunpy as np

cutoff=1e-12

class Loss(object):
    @staticmethod
    def forward(self,y_hat,y):
        raise NotImplementedError('forward function must be implemented')

    @staticmethod
    def backward(self,y_hat,y):
        raise NotImplementedError('backward function must be implemented')

class MeanSquareLoss(Loss):
    """
    y_hat:vector,output from calc
    y:vector,the ground truth
    """
    @staticmethod
    def forward(y_hat,y):
        return 0.5*np.sum(sp.power((y-y_hat),2))

    @staticmethod
    def backward(y_hat,y):
        return y_hat-y

class CrossEntropyLoss(Loss):

    @staticmethod
    def forward(y_hat,y):
        y_hat=_cutoff(y_hat)
        y=_cutoff(y)
        return -np.mean(np.sum(np.nan_to_num(y*np.log(y_hat))+(1-y)*np.log(1-y_hat)),axis=1)

    @staticmethod
    def backward(y_hat,y):
        y_hat=_cutoff(y_hat)
        y=_cutoff(y)
        return (y_hat-y)/(y_hat*(1-y_hat))


# softmax logloss
class LogLikehoodLoss(Loss):

    @staticmethod
    def forward(y_hat,y):
        assert (np.abs(np.sum(y_hat,axis=1)-1.0)<cutoff).all()
        assert (np.abs(np.sum(y,axis=1)-1.0)<cutoff).all()
        y_hat=_cutoff(y_hat)
        y=_cutoff(y)
        return -np.mean(np.sum(np.nan_to_num(y*np.log(y_hat)),axis=1))

    def backward(y_hat,y):
        assert (np.abs(np.sum(y_hat,axis=1)-1.0)<cutoff).all()
        assert (np.abs(np.sum(y,axis=1)-1.0)<cutoff).all()
        y_hat=_cutoff(y_hat)
        y=_cutoff(y)
        return y_hat-y

#to avoid devide zero raise error
def _cutoff(x):
    return np.clip(x,cutoff,1-cutoff)

mse=MSE=MeanSquareLoss
cce=CrossEntropyLoss
mle=LogLikehoodLoss


def get(loss):
    if isinstance(loss,str):
        loss=loss.tolower()
        if loss in ('mse','meansquareLoss'):
            return MeanSquareLoss
        elif loss in ('cce','crossentropyLoss'):
            return CrossEntropyLoss()
        elif loss in ('mle','loglikehoodLoss'):
            return LogLikehoodLoss()
        else:
            raise ValueError('unknown loss name `{}`'.format(loss))
    elif isinstance(loss,Loss):
        return loss
    else:
        raise ValueError('unknown loss type `{}`'.format(loss))

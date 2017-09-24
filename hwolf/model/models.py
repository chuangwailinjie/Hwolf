import numpy as np

from ..base import optimizers
from ..base import losses
from ..layers.core import Softmax

import sys

class Sequential(object):
    def __init__(self):
        self.layers=list()

    def add(self,layer):
        self.layers.append(layer)

    def compile(self,loss,optimizer='adam',**kwargs):
        loss=loss or None
        optimizer=optimizer or None

        self.optimizer=optimizers.get(optimizer)
        self.loss=losses.get(loss)

        #input layer's prev layer is None
        pre_layer=None
        for layer in self.layers:
            layer.connection(pre_layer)
            pre_layer=layer

    def fit(self,X,y,max_iter=100,batch_size=64,shuffle=True,
            validation_split=0.,validation_data=None,verbose=1,file=sys.stdout):

        train_X=X.astype(np.float64) if not np.issubdtype(np.float64,X.dtype) else X
        train_y=y.astype(np.float64) if not np.issubdtype(np.float64,y.dtype) else y

        if 0.<validation_split <1.0:
            #validation_split is tje val_set size
            split=int(len(train_X)*validation_split)
            valid_X,valid_y=train_X[-split:],train_y[-split:]
            train_X,train_y=train_X[:-split],train_y[:-split]

        elif validation_data is not None:
            valid_X,valid_y=validation_data
        else:
            valid_X,valid_y=None,None

        iter_idx=0
        while iter_idx < max_iter:
            iter_idx+=1

            if shuffle:
                seed=np.random.randint(2234)
                np.random.seed(seed)
                np.random.shuffle(train_X)
                np.random.seed(seed)
                np.random.shuffle(train_y)

            #train process
            train_losses,train_predicts,train_targets=[],[],[]
            for b in range(len(train_y)//batch_size):
                batch_begin=b*batch_size
                batch_end=batch_begin+batch_size
                x_batch=train_X[batch_begin:batch_end]
                y_batch=train_y[batch_begin:batch_end]

                y_pred=self.predict(x_batch,is_train=True)
                next_grad=self.loss.backward(y_pred,y_batch)
                for layer in self.layers[::-1]:
                    next_grad=layer.backward(next_grad)

                params=list()
                grads=list()

                #params and grad is layer's parameter
                for layer in self.layers:
                    params+=layer.params
                    grads+=layer.grads

                #params is [W,b]
                #update parameters
                self.optimizer.minimize(params,grads)

                train_losses.append(self.loss.forward(y_pred,y_batch))
                train_predicts.extend(y_pred)
                train_targets.extend(y_batch)

                if verbose == 2:
                    runout = "iter %d, batch %d, train-[loss %.4f, acc %.4f]; " % (
                        iter_idx, b + 1, float(np.mean(train_losses)),
                        float(self.accuracy(train_predicts, train_targets)))
                    print(runout, file=file)

            runout = "iter %d, train-[loss %.4f, acc %.4f]; " % (
                iter_idx, float(np.mean(train_losses)),
                float(self.accuracy(train_predicts, train_targets)))


            if valid_X is not None and valid_y is not None:
                # valid, use updated parameters predict
                valid_losses, valid_predicts, valid_targets = [], [], []
                for b in range(valid_X.shape[0] // batch_size):
                    batch_begin = b * batch_size
                    batch_end = batch_begin + batch_size
                    x_batch = valid_X[batch_begin:batch_end]
                    y_batch = valid_y[batch_begin:batch_end]

                    # forward propagation
                    y_pred = self.predict(x_batch, is_train=False)

                    # got loss and predict
                    valid_losses.append(self.loss.forward(y_pred, y_batch))
                    valid_predicts.extend(y_pred)
                    valid_targets.extend(y_batch)

                # output valid status
                runout += "valid-[loss %.4f, acc %.4f]; " % (
                    float(np.mean(valid_losses)), float(self.accuracy(valid_predicts, valid_targets)))

            if verbose > 0:
                print(runout, file=file)


    def predict(self,X,is_train=False):
        x_next=X
        for layer in self.layers[:]:
            x_next=layer.forward(x_next,is_train=is_train)
        y_pred=x_next
        return y_pred

    def accuracy(self,outputs,targets):
        y_predicts=np.argmax(outputs,axis=1)
        y_targets=np.argmax(targets,axis=1)
        acc= y_predicts==y_targets
        return np.mean(acc)


class Model(object):
    def __init__(self,input,output):
        self.input=input
        self.output=output

    def compile(self,loss,optimizer='sgd',**kwargs):
        loss = loss or None
        optimizer = optimizer or None
        self.optimizer = optimizers.get(optimizer)
        self.loss = losses.get(loss)

    def fit(self,X,y,max_iter=100,batch_size=64,shuffle=True,
            validation_split=0.,validation_data=None,verbose=1,file=sys.stdout):

        train_X = X.astype(np.float64) if not np.issubdtype(np.float64, X.dtype) else X
        train_y = y.astype(np.float64) if not np.issubdtype(np.float64, y.dtype) else y

        if 1. > validation_split > 0.:
            split = int(train_y.shape[0] * validation_split)
            valid_X, valid_y = train_X[-split:], train_y[-split:]
            train_X, train_y = train_X[:-split], train_y[:-split]
        elif validation_data is not None:
            valid_X, valid_y = validation_data
        else:
            valid_X, valid_y = None, None

        iter_idx = 0
        while iter_idx < max_iter:
            iter_idx += 1

            # shuffle
            if shuffle:
                seed = np.random.randint(3457)
                np.random.seed(seed)
                np.random.shuffle(train_X)
                np.random.seed(seed)
                np.random.shuffle(train_y)

            # train
            train_losses, train_predicts, train_targets = [], [], []
            for b in range(train_y.shape[0] // batch_size):
                batch_begin = b * batch_size
                batch_end = batch_begin + batch_size
                x_batch = train_X[batch_begin:batch_end]
                y_batch = train_y[batch_begin:batch_end]

                # forward propagation
                y_pred = self.predict(x_batch, is_train=True)

                # backward propagation
                next_grad = self.loss.backward(y_pred, y_batch)
                pre_layer = self.output
                while pre_layer is not None:
                    next_grad = pre_layer.backward(next_grad)
                    pre_layer = pre_layer.pre_layer

                # get parameter and gradients
                params = list()
                grads = list()
                pre_layer = self.output
                while pre_layer is not None:
                    params += pre_layer.params
                    grads += pre_layer.grads
                    pre_layer = pre_layer.pre_layer

                # update parameters
                self.optimizer.minimize(params, grads)

                # got loss and predict
                train_losses.append(self.loss.forward(y_pred, y_batch))
                train_predicts.extend(y_pred)
                train_targets.extend(y_batch)
                if verbose == 2:
                    runout = "iter %d, batch %d, train-[loss %.4f, acc %.4f]; " % (
                        iter_idx, b + 1, float(np.mean(train_losses)),
                        float(self.accuracy(train_predicts, train_targets)))
                    print(runout, file=file)

            # output train status
            runout = "iter %d, train-[loss %.4f, acc %.4f]; " % (
                iter_idx, float(np.mean(train_losses)),
                float(self.accuracy(train_predicts, train_targets)))

            if valid_X is not None and valid_y is not None:
                # valid
                valid_losses, valid_predicts, valid_targets = [], [], []
                for b in range(valid_X.shape[0] // batch_size):
                    batch_begin = b * batch_size
                    batch_end = batch_begin + batch_size
                    x_batch = valid_X[batch_begin:batch_end]
                    y_batch = valid_y[batch_begin:batch_end]

                    # forward propagation
                    y_pred = self.predict(x_batch, is_train=False)

                    # got loss and predict
                    valid_losses.append(self.loss.forward(y_pred, y_batch))
                    valid_predicts.extend(y_pred)
                    valid_targets.extend(y_batch)

                # output valid status
                runout += "valid-[loss %.4f, acc %.4f]; " % (
                    float(np.mean(valid_losses)), float(self.accuracy(valid_predicts, valid_targets)))

            if verbose > 0:
                print(runout, file=file)

    def predict(self, X, is_train=False):
        """ Calculate an output Y for the given input X. """
        layer = self.input
        X_next = X
        while layer is not None:
            X_next = layer.forward(X_next)
            layer = layer.next_layer
        return X_next

    def accuracy(self, outputs, targets):
        y_predicts = np.argmax(outputs, axis=1)
        y_targets = np.argmax(targets, axis=1)
        acc = y_predicts == y_targets
        return np.mean(acc)

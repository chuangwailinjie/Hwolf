import numpy as np

from layer import Layer

class MaxPoolingLayer(Layer):

    def __init__(self,window_shape,input_shape=None,stride,zero_padding=0):
        super().__init__()
        self.input_shape=input_shape
        if isinstance(zero_padding,int):
            zero_padding=(zero_padding,zero_padding)
        self.zero_padding=zero_padding
        if isinstance(window_shape,int):
            window_shape=(window_shape,window_shape)
        self.window_shape=window_shape
        if isinstance(stride,int):
            stride=(stride,stride)
        self.stride=stride
        self.__delta=None
        if self.input_shape is not None:
            self.connection(None)

    @property
    def delta(self):
        return self.__delta

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def connection(self,pre_layer):
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('the first layer must have input')

        else:
            self.input_shape=pre_layer.output_shape
            self.pre_layer=pre_layer
            pre_layr.next_layer=self

        output_width=(self.input_shape[2]+2*self.zero_padding[1]
                        -self.window_shape[1])//self.stride[1]+1
        output_height=(self.input_shape[1]+2*self.zero_padding[0]
                        -self.window_shape[0])//self.stride[0]+1
        self.output_shape=[self.input_shape[0],output_width,output_height,
                            1 if len(self.input_shape)==3 else self.input_shape[3]]

    def padding(self.inputs,zero_padding):
        inputs=np.asarray(inputs)
        if list(zero_padding)==[0,0]:
            return inputs

        if self.ndim==3:
            #there will teansform 3-D to 4-D
            inputs=inputs[:,:,:,None]

        if self.ndim==4:
            num,input_height,input_width,channel=inputs.shape
            padded_input=np.zeros_like([num,input_width+2*self.zero_padding[0]
                                        input_height+2*self.zero_padding[1],channel])
            padded_input[:,zero_padding[0]:input_height+zero_padding[0],
                            zero_padding[1]:input_width+self.zero_padding[1],:]=inputs
            return padded_input

    def forward(self,inputs,*args,**kwargs):
        inputs=np.asarray(inputs)
        assert list(self.input_shape[1:])==list(inputs.shape[1:])
        self.input_shape=inputs.shape
        self.output_shape[0]=inputs.shape[0]

        if inputs.ndim==3:
            inputs=inputs[:,:,:,None]

        if inputs.ndim==4:
            #num_sample,input_width,input_height,channel
            self.inputs=inputs
        else:
            raise ValueError('input must be the 2-D or 3-D')

        if len(self.output_shape)==3:
            self.output_shape=np.zeros(list(self.output_shape)+[1])
        else:
            self.output_shape=np.zeros(self.output_shape)

        self.padded_input=self.padding(self.inputs,self.zero_padding)

        #the index of max value from window_shape, 2-D,x-y
        self.max_ind=np.zeros(list(self.output_shape)+[2],dtype=int)

        for idx_c in range(self.inputs.shape[3]):#channel
            for bn in range(self.output.shape[0]):#num_sample
                #wb/hb is the begin index,and he/we is size
                wb=hb=0
                he=self.window_shape[0]
                we=self.window_shape[1]
                for i in range(self.output_shape[1]):
                    for j in range(self.output_shape[2]):
                        self.output[bn,i,j,idx_c]=np.max(self.padded_input[bn,hb:he,wb:we,idx_c])
                        #max_ind is the location of max value of current padded_input window
                        max_ind=np.argmax(self.padded_input[bn,hb:he,wb:we,idx_c])
                        max_x,max_y=max_ind/self.window_shape[0],max_ind%self.window_shape
                        self.max_ind[bn,i,j,idx_c]=[max_x+wb,max_y+hb]
                        wb+=self.stride[1]
                        we+=self.stride[1]
                    hb+=self.stride[0]
                    he+=self.stride[0]
                    wb=0
                    we=self.window_shape[1]
        if len(self.output_shape)==3:
            return self.output[:,:,:,0]
        else:
            return self.output

    def backward(self,pre_delta,*args,**kwargs):
        if len(pre_delta)==3:
            self.__delta=np.zeros(tuple(self.input_shape)+(1))
        else:
            self.__delta=np.zeros(self.input_shape)

        for idx_c in range(self.inputs.shape[3]):
            for bn in range(self.output_shape[0]):
                for i in range(self.output_shape[1]):
                    for j in range(self.output_shape[2]):
                        x,y=self.max_ind[bn,i,j,idx_c]
                        #remove influence of zero pad
                        if x<self.zero_padding[0] or x>=self.input_shape[1]+self.zero_padding[0]:
                            continue
                        if y<self.zero_padding[1] or y>=self.input_shape[2]+self.zero_padding[1]:
                            continue
                        x-=self.zero_padding[0]
                        y-=self.zero_padding[1]
                        # only the max value index plus pre_delat,others are zeros not change
                        self._delta[bn,x,y,idx_c]+=pre_delta[bn,i,j,idx_c]

        if len(self.input_shape)==3:
            return self.__delta[:,:,:,0]
        else retuen self.__delta


class AvgPoolingLayer(Layer):
    def __init__(self,window_shape,input_shape=None,stride=1,zero_padding=0):
        super().__init__()
        self.input_shape=input_shape
        if isinstance(zero_padding,int):
            zero_padding=(zero_padding,zero_padding)
        self.zero_padding=zero_padding
        if isinstance(window_shape,int):
            window_shape=(window_shape,window_shape)
        self.window_shape=window_shape
        if isinstance(stride,int):
            stride=(stride,stride)
        self.stride=stride
        self.__delta=None
        if self.input_shape is not None:
            self.connection(None)

    @property
    def delta(self):
        return self.__delta

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def connection(self,pre_layer):
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('the first layer must have input')

        else:
            self.input_shape=pre_layer.output_shape
            self.pre_layer=pre_layer
            pre_layr.next_layer=self

        output_width=(self.input_shape[2]+2*self.zero_padding[1]
                        -self.window_shape[1])//self.stride[1]+1
        output_height=(self.input_shape[1]+2*self.zero_padding[0]
                        -self.window_shape[0])//self.stride[0]+1
        self.output_shape=[self.input_shape[0],output_width,output_height,
                            1 if len(self.input_shape)==3 else self.input_shape[3]]

    def padding(self.inputs,zero_padding):
        inputs=np.asarray(inputs)
        if list(zero_padding)==[0,0]:
            return inputs

        if self.ndim==3:
            #there will teansform 3-D to 4-D
            inputs=inputs[:,:,:,None]

        if self.ndim==4:
            num,input_height,input_width,channel=inputs.shape
            padded_input=np.zeros_like([num,input_width+2*self.zero_padding[0]
                                        input_height+2*self.zero_padding[1],channel])
            padded_input[:,zero_padding[0]:input_height+zero_padding[0],
                            zero_padding[1]:input_width+self.zero_padding[1],:]=inputs
            return padded_input


    def forward(self,inputs,*args,**kwargs):
        inputs=np.asarray(inputs)

        assert list(self.input_shape[1:])==list(inputs.shape[1:])
        self.input_shape=inputs.shape
        self.output_shape[0]=inputs.shape[0]

        if inputs.ndim==3:
            inputs=inputs[:,:,:,None]
        if inputs.ndim==4:
            self.inputs=inputs
        else:
            raise ValueError('.......')

        if len(self.output_shape)==3:
            self.output=np.zeros(list(self.output)+[1])
        else:
            self.output=np.zeros(self.output)

        self.padded_input=self.padding(self.inputs,self.zero_padding)

        for idx_c in range(self.inputs.shape[3]):
            for bn in range(self.inputs.shape[0]):
                wb=hb=0
                he=self.window_shape[0]
                we=self.window_shape[1]
                for i in range(self.output.shape[1]):
                    for j in range(self.output.shape[2]):
                        self.output[bn,i,j,idx_c] = np.sum(self.padded_input[bn,hb:he,wb:we,idx_c])\
                                            / float(np.prod(self.window_shape))
                        wb+=self.stride[0]
                        we+=self.stride[1]
                    hb+=self.stride[0]
                    he+=self.stride[1]
                    wb=0
                    we=self.window_shape[1]
        if len(self.output_shape)==3:
            return self.output[:,:,:,0]
        else:
            reutrn self.output



    def backward(self,pre_delta,*args,**kwargs):
        if len(self.input_shape) == 3:
            self.__delta = np.zeros(tuple(self.input_shape) + (1, ))
        else:
            self.__delta = np.zeros(self.inputs.shape)
        for idx_c in xrange(self.inputs.shape[3]):
            for bn in xrange(self.inputs.shape[0]):
                wb = hb = 0
                he = self.window_shape[0]
                we = self.window_shape[1]
                for i in xrange(self.output.shape[1]):
                    for j in xrange(self.output.shape[2]):
                        self.__delta[bn,hb:he,wb:we,idx_c] += (pre_delta[bn,i,j,idx_c] \
                            / float(np.prod(self.window_shape)))
                        wb += self.stride[1]
                        we += self.stride[1]
                    hb += self.stride[0]
                    he += self.stride[0]
                    wb = 0; we = self.window_shape[1]
        if len(self.input_shape) == 3:
            return self.__delta[:,:,:,0]
        else:
            return self.__delta

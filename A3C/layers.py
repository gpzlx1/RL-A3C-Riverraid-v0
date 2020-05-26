import torch 
import numpy as np
import torch.nn.functional as F

class Layer(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError 

    def load_weights(self):
        raise NotImplementedError 

class Conv2d(Layer):

    def __init__(self,in_channels, out_channels, kernel_size,stride=1,
                 padding=0,bias=True,groups = 1,dilation = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,kernel_size)
        self.stride = (stride,stride)
        self.padding = (padding,padding)
        self.groups = groups
        self.dilation = dilation
        if bias:
            self.bias = torch.Tensor(out_channels)

    def init_weight(self,random = True):
        if random:
            pass
        else:
            self.weight = torch.Tensor(self.in_channels,self.out_channels // self.groups, *self.kernel_size)
    def load_weight(self,weight):
        self.weight = weight

    def forward(self,input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def backward(self):
        pass

class LSTMCell(Layer):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        '''初始化所有参数为0'''
        self.weight_ih = torch.zeros(4 * hidden_size, input_size)
        self.weight_hh = torch.zeros(4 * hidden_size, hidden_size)
        if bias:
            self.bias_ih = torch.zeros(4 * hidden_size)
            self.bias_hh = torch.zeros(4 * hidden_size)
        else:
            self.bias_ih = None
            self.bias_hh = None
        
    def init_weight(self, random=True, loc=0.0, scale=1):
        if random:
            self.weight_ih = torch.Tensor(np.random.normal(loc=0.0, scale=1, size=self.weight_ih.shape))
            self.weight_hh = torch.Tensor(np.random.normal(loc=0.0, scale=1, size=self.weight_hh.shape))
        else:
            self.weight_ih = torch.zeros(4 * hidden_size, input_size)
            self.weight_hh = torch.zeros(4 * hidden_size, hidden_size)

    def init_bias(self, random=True, loc=0.0, scale=1):
        if self.bias is not None:
            return 

        if random:
            self.bias_ih = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.bias_ih.shape))
            self.bias_hh = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.bias_hh.shape))
        else:
            self.bias_ih = torch.zeros(4 * hidden_size)
            self.bias_hh = torch.zeros(4 * hidden_size)

    def load_weights(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

if __name__ == "__main__":
    LSTM = LSTMCell(32 * 3 * 3, 256)
    print(LSTM.weight_ih)
    print(LSTM.bias_hh)
    LSTM.init_weight()
    print(LSTM.weight_ih)

    conv_1 = Conv2d(1, 1, 3, stride=2, padding=1)
    conv_1.init_weight(False)
    print(conv_1.weight)


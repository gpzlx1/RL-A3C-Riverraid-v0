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
        self.dilation = (dilation,dilation)
        if bias:
            self.bias = torch.Tensor(out_channels)
        else:
            self.bias = torch.zeros(out_channels)

    def init_weight(self,random = True):
        if random:
            pass
        else:
            self.weight = torch.Tensor(self.out_channels,self.in_channels // self.groups, *self.kernel_size)

    def load_weights(self,weight):
        self.weight = weight

    def forward(self,input):
        self.input = input
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def backward(self,top_grad):
        #only for in_channel = 1
        top_grad = top_grad.squeeze(0).unsqueeze(1)
        conv_for_back = Conv2d(self.in_channels,self.out_channels,top_grad.shape[2],padding = 1,bias = False,dilation = self.stride[0])
        conv_for_back.load_weights(top_grad)
        weight_grad = conv_for_back.forward(self.input).squeeze(0).unsqueeze(1)
        bias_grad = torch.ones(self.out_channels) * top_grad.shape[2] * top_grad.shape[2]
        if (self.input.shape[2]-top_grad.shape[2])%2 == 0:
            return weight_grad,bias_grad
        else:
            #t = torch.zeros(ret.shape)
            t = weight_grad[:,:,:self.weight.shape[2],:self.weight.shape[2]]
            return t,bias_grad

class Linear(Layer):

    def __init__(self,in_size,out_size,bias = True):
        self.in_size = in_size
        self.out_size = out_size
        if bias:
            self.bias = torch.Tensor(out_size)
        else:
            self.bias = torch.zeros(out_size)

    def init_weight(self,random = False):
        if random:
            self.weight = torch.Tensor(self.out_size,self.in_size)
        else:
            self.weight = torch.zeros(self.out_size,self.in_size)

    def load_weights(self,weight):
        self.weight = weight

    def forward(self,input):
        self.input = input
        return F.linear(input,self.weight,self.bias)

    def backward(self,top_grad):
        top_grad = top_grad.squeeze(0).unsqueeze(1)
        weight_grad = torch.matmul(top_grad,self.input)
        bias_grad = torch.ones(self.out_size)
        return weight_grad,bias_grad


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

    #conv_test
    input = torch.randn(1,43,43).unsqueeze(0)
    input.requires_grad = True
    kernel_size = (3,3)
    weight = torch.randn(32,1 // 1, *kernel_size)

    conv_1 = Conv2d(1, 32, 3, stride=2, padding=1)
    conv_1.load_weights(weight)

    Convtest = torch.nn.Conv2d(1,32, 3, stride=2, padding=1)
    Convtest.weight.data = weight
    Convtest.bias.data = conv_1.bias

    #forward_test
    result = Convtest(input)
    my_result = conv_1.forward(input)

    #backward_test
    result.sum().backward()
    weight_grad,bias_grad = conv_1.backward(torch.ones(result.shape))
    print(sum(sum(sum(sum(abs(Convtest.weight.grad-weight_grad))))))
    print(Convtest.weight.grad.shape,weight_grad.shape)
    print(sum(abs(Convtest.bias.grad-bias_grad)))
    print(Convtest.bias.grad.shape,bias_grad.shape)

    #linear_test
    input = torch.randn(1,256)
    input.requires_grad = True
    weight = torch.randn(18,256)

    my_linear = Linear(256,18)
    my_linear.load_weights(weight)

    linear = torch.nn.Linear(256, 18)
    linear.weight.data = weight
    linear.bias.data = my_linear.bias

    #forward_test
    result = linear(input)
    my_result = my_linear.forward(input)
    print(sum(result-my_result))

    #backward_test
    result.sum().backward()
    weight_grad,bias_grad = my_linear.backward(torch.ones(result.shape))
    print(sum(sum(weight_grad-linear.weight.grad)))
    print(weight_grad.shape,linear.weight.grad.shape)
    print(sum(bias_grad - linear.bias.grad))
    print(bias_grad.shape,linear.bias.grad.shape)


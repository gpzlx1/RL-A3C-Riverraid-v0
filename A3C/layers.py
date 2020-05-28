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
            self.bias = torch.zeros(out_channels)
        else:
            self.bias = None

    def init_weight(self,random = True, loc=0.0, scale=1):
        if random:
            self.weight = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=(self.out_channels,self.in_channels // self.groups, *self.kernel_size)))
        else:
            self.weight = torch.Tensor(self.out_channels,self.in_channels // self.groups, *self.kernel_size)


    def init_bias(self, random = True, loc=0.0, scale=1):
        if self.bias is None:
            return 

        if random:
            self.bias = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.out_channels))
        else:
            self.bias = torch.zeros(self.out_channels)

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
            self.bias = torch.zeros(out_size)
        else:
            self.bias = None

    def init_weight(self,random = True, loc=0.0, scale=1):
        if random:
            self.weight = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=(self.out_size,self.in_size)))
        else:
            self.weight = torch.zeros(self.out_size,self.in_size)

    
    def init_bias(self, random = True, loc=0.0, scale=1):
        if self.bias is None:
            return 

        if random:
            self.bias = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.out_size))
        else:
            self.bias = torch.zeros(self.out_size)

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

        #中间变量
        self.pre_inputs = None
        self.ingate = None
        self.forgetgate = None
        self.cellgate = None
        self.outgate = None
        self.cy = None
        self.cx = None
        self.hy = None
        self.hx = None
        
        #梯度        
        self.grad_weight_ih = None
        self.grad_weight_hh = None
        self.grad_bias_ih = None
        self.grad_bias_hh = None
        

    def init_weight(self, random=True, loc=0.0, scale=1):
        if random:
            self.weight_ih = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.weight_ih.shape))
            self.weight_hh = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.weight_hh.shape))
        else:
            self.weight_ih = torch.zeros(4 * hidden_size, input_size)
            self.weight_hh = torch.zeros(4 * hidden_size, hidden_size)

    def init_bias(self, random=True, loc=0.0, scale=1):
        if self.bias is False:
            return 

        if random:
            self.bias_ih = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.bias_ih.shape))
            self.bias_hh = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.bias_hh.shape))
        else:
            self.bias_ih = torch.zeros(4 * hidden_size)
            self.bias_hh = torch.zeros(4 * hidden_size)

    def load_weights(self):
        pass

    def forward(self, inputs, hidden):
        # https://medium.com/@andre.holzner/lstm-cells-in-pytorch-fab924a78b1c
        hx, cx = hidden
        self.hx = hx
        self.cx = cx
        self.pre_inputs = inputs
        gates = F.linear(self.pre_inputs, self.weight_ih, self.bias_ih) + F.linear(self.hx, self.weight_hh, self.bias_hh)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        self.ingate = torch.sigmoid(ingate) # i_t 第二个
        self.forgetgate = torch.sigmoid(forgetgate) # f_t 第一个
        self.cellgate = torch.tanh(cellgate) # g_t 第三个
        self.outgate = torch.sigmoid(outgate) # o_t 第四个

        self.cy = (self.forgetgate * self.cx) + (self.ingate * self.cellgate)
        self.hy = self.outgate * torch.tanh(self.cy)
        return self.hy, self.cy
        

    def backward(self, top_grad_h, top_grad_c):
        grad_outgate =  torch.tanh(self.cy) * top_grad_h
        grad_c = (1 - (torch.tanh(self.cy))**2) * self.outgate * top_grad_h

        grad_forgetgate = self.cx * grad_c
        grad_ingate = self.cellgate * grad_c
        grad_cellgate = self.ingate * grad_c

        df_input = self.forgetgate * (1 - self.forgetgate) * grad_forgetgate
        di_input = self.ingate  * (1 - self.ingate) * grad_ingate
        dg_input = (1 - (self.cellgate)**2) * grad_cellgate
        do_input = self.outgate  * (1 - self.outgate) * grad_outgate

        grad_Wif = df_input.t().matmul(self.pre_inputs)
        grad_Whf = df_input.t().matmul(self.hx)
        grad_bf = df_input  

        grad_Wii = di_input.t().matmul(self.pre_inputs)
        grad_Whi = di_input.t().matmul(self.hx)
        grad_bi =  di_input 

        grad_Wig = dg_input.t().matmul(self.pre_inputs)
        grad_Whg = dg_input.t().matmul(self.hx)
        grad_bg = dg_input 

        grad_Wio = do_input.t().matmul(self.pre_inputs)
        grad_Who = do_input.t().matmul(self.hx)
        grad_bo = do_input

        self.grad_weight_ih = torch.cat((grad_Wii, grad_Wif, grad_Wig, grad_Wio))
        self.grad_weight_hh = torch.cat((grad_Whi, grad_Whf, grad_Whg, grad_Who))
        self.grad_bias_hh = torch.cat((grad_bi, grad_bf, grad_bg, grad_bo),1).squeeze(0)
        self.grad_bias_ih = self.grad_bias_hh

        bottom_grad_c = self.forgetgate * grad_c
        
        param_hi, param_hf, param_hg, param_ho = self.weight_hh.chunk(4,0)
        bottom_grad_h = di_input.matmul(param_hi) + df_input.matmul(param_hf) +\
                dg_input.matmul(param_hg) + do_input.matmul(param_ho)

        param_ii, param_if, param_ig, param_io = self.weight_ih.chunk(4,0)
        bottom_grad_inputs = di_input.matmul(param_ii) + df_input.matmul(param_if) +\
                dg_input.matmul(param_ig) + do_input.matmul(param_io)

        return bottom_grad_inputs, bottom_grad_h, bottom_grad_c

class LSTMTest(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMTest, self).__init__()
        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        hx, cx = self.lstm(inputs, (hx, cx))
        return hx, cx


if __name__ == "__main__":
    
    
    print("begin ------------lstm---------------")

    cx = torch.randn((1, 256), requires_grad=True)
    hx = torch.randn((1, 256), requires_grad=True)
    inputs = torch.randn(1,32*3*3, requires_grad=True)


    #标准model
    test = LSTMTest(32 * 3 * 3, 256)
    h1, c1 = test((inputs, (hx,cx)))

    #print(h1)
    #print(c1)
    h1.sum().backward()
    

    LSTM = LSTMCell(32 * 3 * 3, 256)
    LSTM.weight_hh = test.lstm.weight_hh
    LSTM.weight_ih = test.lstm.weight_ih
    LSTM.bias_hh = test.lstm.bias_hh
    LSTM.bias_ih = test.lstm.bias_ih
    print(LSTM.weight_hh.shape)
    print(LSTM.bias_hh.shape)
    h2,c2 = LSTM.forward(inputs, (hx,cx))
    #print(h2)
    #print(c2)
    bottom_grad_inputs, bottom_grad_h, bottom_grad_c = LSTM.backward(torch.ones(h1.shape),None)
    print(sum(sum(abs(test.lstm.weight_hh.grad - LSTM.grad_weight_hh))))
    print(sum(sum(abs(test.lstm.weight_ih.grad - LSTM.grad_weight_ih))))
    print(sum(abs(test.lstm.bias_ih.grad - LSTM.grad_bias_ih)))
    print(sum(abs(test.lstm.bias_hh.grad - LSTM.grad_bias_hh)))
    print(sum(sum(abs(hx.grad - bottom_grad_h))))
    print(sum(sum(abs(cx.grad - bottom_grad_c))))
    print(test.lstm.bias_hh.grad.shape, LSTM.grad_bias_hh.shape)
    print(test.lstm.weight_hh.grad.shape, LSTM.grad_weight_hh.shape)
    print(hx.grad.shape, bottom_grad_h.shape)
    print(cx.grad.shape, bottom_grad_c.shape)
    print(sum(sum(abs(inputs.grad - bottom_grad_inputs))))
    print(inputs.grad.shape, bottom_grad_inputs.shape)

    print("begin ------------conv---------------")

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


    print("begin ------------linear---------------")
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


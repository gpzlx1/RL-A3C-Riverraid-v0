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
        self.grad_pre_inputs = None
        self.grad_ingate = None
        self.grad_forgetgate = None
        self.grad_cellgate = None
        self.grad_outgate = None
        self.grad_cy = None
        self.grad_cx = None
        self.grad_hy = None
        self.grad_hx = None
        
        self.grad_weight_ih = None
        self.grad_weight_hh = None
        self.grad_bias_ih = None
        self.grad_bias_hh = None
        

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
        
        param_i, param_f, param_g, param_o = self.weight_hh.chunk(4,0)
        bottom_grad_h = di_input.matmul(param_i) + df_input.matmul(param_f) +\
                dg_input.matmul(param_g) + do_input.matmul(param_o)

        return bottom_grad_h, bottom_grad_c

class LSTMTest(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMTest, self).__init__()
        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        hx, cx = self.lstm(inputs, (hx, cx))
        return hx, cx


if __name__ == "__main__":
    
        
    cx = torch.randn((1, 256), requires_grad=True)
    hx = torch.randn((1, 256), requires_grad=True)
    inputs = torch.randn(1,32*3*3)


    #标准model
    test = LSTMTest(32 * 3 * 3, 256)
    h1, c1 = test((inputs, (hx,cx)))

    #print(h1)
    #print(c1)
    h1.sum().backward()
    

    LSTM = LSTMCell(32 * 3 * 3, 256)
    LSTM.weight_hh = test.lstm.weight_hh#.copy()
    LSTM.weight_ih = test.lstm.weight_ih
    LSTM.bias_hh = test.lstm.bias_hh
    LSTM.bias_ih = test.lstm.bias_ih
    print(LSTM.weight_hh.shape)
    print(LSTM.bias_hh.shape)
    h2,c2 = LSTM.forward(inputs, (hx,cx))
    #print(h2)
    #print(c2)
    bottom_grad_h, bottom_grad_c = LSTM.backward(torch.ones(h1.shape),None)
    print(sum(sum(abs(test.lstm.weight_hh.grad - LSTM.grad_weight_hh))))
    print(sum(sum(abs(test.lstm.weight_ih.grad - LSTM.grad_weight_ih))))
    print(sum(abs(test.lstm.bias_ih.grad - LSTM.grad_bias_ih)))
    print(sum(abs(test.lstm.bias_hh.grad - LSTM.grad_bias_hh)))
    print(test.lstm.bias_hh.grad.shape, LSTM.grad_bias_hh.shape)
    print(test.lstm.weight_hh.grad.shape, LSTM.grad_weight_hh.shape)
    print(sum(sum(abs(hx.grad - bottom_grad_h))))
    print(hx.grad.shape, bottom_grad_h.shape)
    print(sum(sum(abs(cx.grad - bottom_grad_c))))
    print(cx.grad.shape, bottom_grad_c.shape)



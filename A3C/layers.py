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
        
        #grad
        self.grad_bias = None
        self.grad_weight = None

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

    def backward(self, top_grad_t):
        top_grad = top_grad_t.transpose(0,1)
        conv_for_weight = Conv2d(self.in_channels,self.out_channels,top_grad.shape[2],padding = self.padding[0],bias = False,dilation = self.stride[0])
        conv_for_weight.load_weights(top_grad)
        weight_grad = conv_for_weight.forward(self.input.transpose(0,1)).transpose(0,1)
        self.grad_bias = torch.sum(torch.sum(torch.sum(top_grad,3),2),1)
        conv_for_back = F.conv_transpose2d(top_grad_t,self.weight,torch.zeros(self.in_channels),self.stride,self.padding,(self.input.shape[2]-top_grad.shape[2])%2,
                                           self.groups,self.dilation)
        if (self.input.shape[2]-top_grad.shape[2])%2 == 0:
            self.grad_weight = weight_grad
            return conv_for_back
        else:
            self.grad_weight = weight_grad[:,:,:self.weight.shape[2],:self.weight.shape[2]]
            return conv_for_back[:,:,:self.input.shape[2],:self.input.shape[2]]





class Linear(Layer):

    def __init__(self,in_size,out_size,bias = True):
        self.in_size = in_size
        self.out_size = out_size
        if bias:
            self.bias = torch.zeros(out_size)
        else:
            self.bias = None

        #grad
        self.grad_bias = None
        self.grad_weight = None

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
        top_grad_t = top_grad.t()
        self.grad_weight = torch.matmul(top_grad_t,self.input)
        self.grad_bias = top_grad.squeeze(0)
        return torch.matmul(top_grad,self.weight)


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
        self.pre_inputs = []
        self.ingate = []
        self.forgetgate = []
        self.cellgate = []
        self.outgate = []
        self.cy = []
        self.cx = []
        self.hy = []
        self.hx = []

        #梯度
        self.grad_weight_ih = []
        self.grad_weight_hh = []
        self.grad_bias_ih = []
        self.grad_bias_hh = []


    def init_weight(self, random=True, loc=0.0, scale=1):
        if random:
            self.weight_ih = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.weight_ih.shape))
            self.weight_hh = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.weight_hh.shape))
        else:
            self.weight_ih = torch.zeros(4 * self.hidden_size, self.input_size)
            self.weight_hh = torch.zeros(4 * self.hidden_size, self.hidden_size)

    def init_bias(self, random=True, loc=0.0, scale=1):
        if self.bias is False:
            return 

        if random:
            self.bias_ih = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.bias_ih.shape))
            self.bias_hh = torch.Tensor(np.random.normal(loc=loc, scale=scale, size=self.bias_hh.shape))
        else:
            self.bias_ih = torch.zeros(4 * self.hidden_size)
            self.bias_hh = torch.zeros(4 * self.hidden_size)

    def load_weights(self):
        pass

    def forward(self, inputs, hidden):
        # https://medium.com/@andre.holzner/lstm-cells-in-pytorch-fab924a78b1c
        hx, cx = hidden

        self.hx.append(hx)
        self.cx.append(cx)
        self.pre_inputs.append(inputs)

        gates = F.linear(inputs, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate) # i_t 第二个
        forgetgate = torch.sigmoid(forgetgate) # f_t 第一个
        cellgate = torch.tanh(cellgate) # g_t 第三个
        outgate = torch.sigmoid(outgate) # o_t 第四个


        self.ingate.append(ingate)
        self.forgetgate.append(forgetgate)
        self.cellgate.append(cellgate)
        self.outgate.append(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        self.cy.append(cy)
        self.hy.append(hy)
        return hy, cy


    def backward(self, top_grad_h, top_grad_c):
        bottom_grad_inputs = []
        bottom_grad_h = []
        bottom_grad_c = []
        for i in range(len(top_grad_h)):
            grad_outgate =  torch.tanh(self.cy[i]) * top_grad_h[i]
            grad_c = (1 - (torch.tanh(self.cy[i]))**2) * self.outgate[i] * top_grad_h[i]

            grad_forgetgate = self.cx[i] * grad_c
            grad_ingate = self.cellgate[i] * grad_c
            grad_cellgate = self.ingate[i] * grad_c

            df_input = self.forgetgate[i] * (1 - self.forgetgate[i]) * grad_forgetgate
            di_input = self.ingate[i]  * (1 - self.ingate[i]) * grad_ingate
            dg_input = (1 - (self.cellgate[i])**2) * grad_cellgate
            do_input = self.outgate[i]  * (1 - self.outgate[i]) * grad_outgate

            grad_Wif = df_input.t().matmul(self.pre_inputs[i])
            grad_Whf = df_input.t().matmul(self.hx[i])
            grad_bf = df_input

            grad_Wii = di_input.t().matmul(self.pre_inputs[i])
            grad_Whi = di_input.t().matmul(self.hx[i])
            grad_bi =  di_input

            grad_Wig = dg_input.t().matmul(self.pre_inputs[i])
            grad_Whg = dg_input.t().matmul(self.hx[i])
            grad_bg = dg_input

            grad_Wio = do_input.t().matmul(self.pre_inputs[i])
            grad_Who = do_input.t().matmul(self.hx[i])
            grad_bo = do_input

            self.grad_weight_ih.append(torch.cat((grad_Wii, grad_Wif, grad_Wig, grad_Wio)))
            self.grad_weight_hh.append(torch.cat((grad_Whi, grad_Whf, grad_Whg, grad_Who)))
            self.grad_bias_hh.append(torch.cat((grad_bi, grad_bf, grad_bg, grad_bo),1).squeeze(0))
            self.grad_bias_ih = self.grad_bias_hh

            bottom_grad_c.append( self.forgetgate[i].mul(grad_c) )

            param_hi, param_hf, param_hg, param_ho = self.weight_hh.chunk(4,0)
            bottom_grad_h.append( di_input.matmul(param_hi) + df_input.matmul(param_hf) +\
                    dg_input.matmul(param_hg) + do_input.matmul(param_ho) )

            param_ii, param_if, param_ig, param_io = self.weight_ih.chunk(4,0)
            bottom_grad_inputs.append( di_input.matmul(param_ii) + df_input.matmul(param_if) +\
                    dg_input.matmul(param_ig) + do_input.matmul(param_io) )

        return bottom_grad_inputs, bottom_grad_h, bottom_grad_c

    def clear_grad(self):
               #中间变量
        self.pre_inputs.clear()
        self.ingate.clear()
        self.forgetgate.clear()
        self.cellgate.clear()
        self.outgate.clear()
        self.cy.clear()
        self.cx.clear()
        self.hy.clear()
        self.hx.clear()

        #梯度
        self.grad_weight_ih.clear()
        self.grad_weight_hh.clear()
        self.grad_bias_ih.clear()
        self.grad_bias_hh.clear()

class LSTMTest(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMTest, self).__init__()
        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        hx, cx = self.lstm(inputs, (hx, cx))
        return hx, cx


if __name__ == "__main__":


    
    def eval(value1, value2):
        if value1.shape != value2.shape:
            print("error")
            return 
        
        print(torch.max(torch.abs(value1 / value2)), torch.min(torch.abs(value1 / value2)))
        
    print("begin ------------lstm---------------")

    cx = torch.randn((1, 256), requires_grad=True)
    hx = torch.randn((1, 256), requires_grad=True)
    input1 = torch.randn(1,32*3*3, requires_grad=True)
    input2 = torch.randn(1,32*3*3, requires_grad=True)


    print("---one element test -----")
    #标准model
    test1 = LSTMTest(32 * 3 * 3, 256)
    h1, c1 = test1((input1, (hx,cx)))

    #print(h1)
    #print(c1)
    h1.sum().backward()


    LSTM = LSTMCell(32 * 3 * 3, 256)
    LSTM.weight_hh = test1.lstm.weight_hh.data
    LSTM.weight_ih = test1.lstm.weight_ih.data
    LSTM.bias_hh = test1.lstm.bias_hh.data
    LSTM.bias_ih = test1.lstm.bias_ih.data
    
    
    h2,c2 = LSTM.forward(input1, (hx,cx))
    #print(h2)
    #print(c2)
    eval(h1,h2)
    eval(c1,c2)
    bottom_grad_inputs, bottom_grad_h, bottom_grad_c = LSTM.backward([torch.ones(h1.shape)],None)
    eval(test1.lstm.weight_hh.grad , LSTM.grad_weight_hh[0])
    eval(test1.lstm.weight_ih.grad , LSTM.grad_weight_ih[0])
    eval(test1.lstm.bias_ih.grad , LSTM.grad_bias_ih[0])
    eval(test1.lstm.bias_hh.grad , LSTM.grad_bias_hh[0])
    eval(hx.grad , bottom_grad_h[0])
    eval(cx.grad , bottom_grad_c[0])
    print()
    eval(input1.grad , bottom_grad_inputs[0])
    
    print("---two element test -----")

    cx = torch.randn((1, 256), requires_grad=True)
    hx = torch.randn((1, 256), requires_grad=True)
    input1 = torch.randn(1,32*3*3, requires_grad=True)
    input2 = torch.randn(1,32*3*3, requires_grad=True)

    test1 = LSTMTest(32 * 3 * 3, 256)
    h1, c1 = test1((input1, (hx,cx)))
    h2, c2 = test1((input2, (hx,cx)))
    #print(h1)
    #print(c1)
    (h1 + h2).sum().backward()


    LSTM.clear_grad()
    LSTM.weight_hh = test1.lstm.weight_hh.data
    LSTM.weight_ih = test1.lstm.weight_ih.data
    LSTM.bias_hh = test1.lstm.bias_hh.data
    LSTM.bias_ih = test1.lstm.bias_ih.data

    h3,c3 = LSTM.forward(input1, (hx,cx))
    h4,c4 = LSTM.forward(input2, (hx,cx))
    bottom_grad_inputs, bottom_grad_h, bottom_grad_c = LSTM.backward([torch.ones(h3.shape), torch.ones(h4.shape)],None)
    

    eval(h1,h3)
    eval(c1,c3)
    eval(h2,h4)
    eval(c2,c4)
    eval(test1.lstm.weight_hh.grad , sum(LSTM.grad_weight_hh))
    eval(test1.lstm.weight_ih.grad , sum(LSTM.grad_weight_ih))
    eval(test1.lstm.bias_ih.grad , sum(LSTM.grad_bias_ih))
    eval(test1.lstm.bias_hh.grad , sum(LSTM.grad_bias_hh))
    eval(hx.grad , sum(bottom_grad_h))
    eval(cx.grad , sum(bottom_grad_c))
    print()
    eval(input1.grad , bottom_grad_inputs[0])
    eval(input2.grad , bottom_grad_inputs[1])
    '''
    print("begin ------------conv---------------")

    #conv_test
    input = torch.randn(32,6,6).unsqueeze(0)
    input.requires_grad = True


    conv_1 = Conv2d(32, 32, 3, stride=2, padding=1)
   

    Convtest = torch.nn.Conv2d(32,32, 3, stride=2, padding=1)
    conv_1.weight = Convtest.weight.data
    conv_1.bias = Convtest.bias.data

    #forward_test
    result = Convtest(input)
    my_result = conv_1.forward(input)

    #backward_test
    result.sum().backward()
    print("result", result.shape)
    bottom_grad = conv_1.backward(torch.ones(result.shape))
    print(sum(sum(sum(sum(abs(Convtest.weight.grad-conv_1.grad_weight))))))
    print(Convtest.weight.grad.shape,conv_1.grad_weight.shape)
    print(sum(abs(Convtest.bias.grad-conv_1.grad_bias)))
    print(Convtest.bias.grad.shape,conv_1.grad_bias.shape)

    print(sum(sum(sum(sum(abs(bottom_grad - input.grad))))))
    print(bottom_grad.shape ,input.grad.shape)

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
    bottom_grad = my_linear.backward(torch.ones(result.shape))

    print(sum(sum(my_linear.grad_weight-linear.weight.grad)))
    print(my_linear.grad_weight.shape,linear.weight.grad.shape)
    print(sum(my_linear.grad_bias - linear.bias.grad))
    print(my_linear.grad_bias.shape,linear.bias.grad.shape)

    print(sum(sum(abs(bottom_grad-input.grad))))
    print(bottom_grad.shape, input.grad.shape)
    '''
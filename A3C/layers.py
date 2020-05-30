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
        self.grad_bias = []
        self.grad_weight = []
        self.input = []

    def clear_grad(self):
        self.grad_bias.clear()
        self.grad_weight.clear()
        self.input.clear()

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
        self.input.append(input)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def backward(self, top_grad_list):
        all_conv_for_back = []
        for i in range(len(top_grad_list)):
            top_grad_t = top_grad_list[i]
            top_grad = top_grad_t.transpose(0,1)
            conv_for_weight = Conv2d(self.in_channels,self.out_channels,top_grad.shape[2],padding = self.padding[0],bias = False,dilation = self.stride[0])
            conv_for_weight.load_weights(top_grad)
            weight_grad = conv_for_weight.forward(self.input[i].transpose(0,1)).transpose(0,1)
            self.grad_bias.append(torch.sum(torch.sum(torch.sum(top_grad,3),2),1))
            conv_for_back = F.conv_transpose2d(top_grad_t,self.weight,torch.zeros(self.in_channels),self.stride,self.padding,(self.input[i].shape[2]-top_grad.shape[2])%2,
                                               self.groups,self.dilation)
            if (self.input[i].shape[2]-top_grad.shape[2])%2 == 0:
                self.grad_weight.append(weight_grad)
                all_conv_for_back.append(conv_for_back)
            else:
                self.grad_weight.append(weight_grad[:,:,:self.weight.shape[2],:self.weight.shape[2]])
                all_conv_for_back.append(conv_for_back[:,:,:self.input[i].shape[2],:self.input[i].shape[2]])
        return all_conv_for_back




class Linear(Layer):

    def __init__(self,in_size,out_size,bias = True):
        self.in_size = in_size
        self.out_size = out_size
        if bias:
            self.bias = torch.zeros(out_size)
        else:
            self.bias = None

        #grad
        self.grad_bias = []
        self.grad_weight = []
        self.input = []

    def clear_grad(self):
        self.grad_weight.clear()
        self.grad_bias.clear()
        self.input.clear()

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
        self.input.append(input)
        return F.linear(input,self.weight,self.bias)

    def backward(self,top_grad_list):
        ret = []
        for i in range(len(top_grad_list)):
            top_grad = top_grad_list[i]
            top_grad_t = top_grad.t()
            self.grad_weight.append(torch.matmul(top_grad_t,self.input[i]))
            self.grad_bias.append(top_grad.squeeze(0))
            ret.append(torch.matmul(top_grad,self.weight))
        return ret


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

        cy = forgetgate.mul(cx).add(ingate.mul(cellgate))
        hy = outgate.mul(torch.tanh(cy))

        self.cy.append(cy)
        self.hy.append(hy)
        return hy, cy


    def backward(self, top_grad_h, top_grad_c):
        bottom_grad_inputs = []
        bottom_grad_c = 0
        bottom_grad_h = 0
        for i in reversed(range(len(top_grad_h))):
            top_grad_h_this_iteration = top_grad_h[i] + bottom_grad_h
            grad_outgate = torch.tanh(self.cy[i]).mul(top_grad_h_this_iteration)
            temp = torch.tanh(self.cy[i])
            grad_c = (1 - temp.mul(temp)).mul(self.outgate[i]).mul(top_grad_h_this_iteration) + top_grad_c[i] + bottom_grad_c

            grad_forgetgate = self.cx[i].mul(grad_c)
            grad_ingate = self.cellgate[i].mul(grad_c)
            grad_cellgate = self.ingate[i].mul(grad_c)

            di_input = self.ingate[i].mul(1 - self.ingate[i]).mul(grad_ingate)
            df_input = self.forgetgate[i].mul(1 - self.forgetgate[i]).mul(grad_forgetgate)
            dg_input = (1 - self.cellgate[i].mul(self.cellgate[i])).mul(grad_cellgate)
            do_input = self.outgate[i].mul(1 - self.outgate[i]).mul(grad_outgate)

            d_ifgo = torch.cat((di_input,df_input,dg_input,do_input),1)
            grad_weight_ih = d_ifgo.t().matmul(self.pre_inputs[i])
            grad_weight_hh = d_ifgo.t().matmul(self.hx[i])
           
            self.grad_weight_ih.append(grad_weight_ih)
            self.grad_weight_hh.append(grad_weight_hh)
            self.grad_bias_hh.append(d_ifgo.squeeze(0))
            self.grad_bias_ih = self.grad_bias_hh

            param_ii, param_if, param_ig, param_io = self.weight_ih.chunk(4,0)
            bottom_grad_inputs.append( di_input.matmul(param_ii).add(df_input.matmul(param_if)).add( \
                    dg_input.matmul(param_ig)).add(do_input.matmul(param_io)) )

            
            bottom_grad_c = self.forgetgate[i] * grad_c
            param_hi, param_hf, param_hg, param_ho = self.weight_hh.chunk(4,0)

            bottom_grad_h = di_input.matmul(param_hi) + df_input.matmul(param_hf) +\
                                    dg_input.matmul(param_hg) + do_input.matmul(param_ho)

        return bottom_grad_inputs.reverse()

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
        
        print(torch.max(torch.abs(value1 / value2)), torch.min(torch.abs(value1 / value2)), torch.mean(torch.abs(value1 / value2)))
        
    print("begin ------------lstm---------------")


    test1 = LSTMTest(32 * 3 * 3, 256)
    LSTM = LSTMCell(32 * 3 * 3, 256)
    LSTM.weight_hh = test1.lstm.weight_hh.data
    LSTM.weight_ih = test1.lstm.weight_ih.data
    LSTM.bias_hh = test1.lstm.bias_hh.data
    LSTM.bias_ih = test1.lstm.bias_ih.data
    LSTM.clear_grad()

    print("---many elements test -----")
    loss = 0
    top_grad_h = []
    top_grad_c = []

    cx = torch.randn((1, 256))
    hx = torch.randn((1, 256))

    for i in range(200):
        #hx = hx.detach()
        #cx = cx.detach()
        inputs = torch.randn(1,32*3*3)
        mh,mc = LSTM.forward(inputs, (hx,cx))
        hx, cx = test1((inputs, (hx,cx)))
        
        loss = loss + hx.sum()

        
        top_grad_h.append(torch.ones(mh.shape))
        top_grad_c.append(torch.zeros(mc.shape))

    loss.backward()
    bottom_grad_inputs = LSTM.backward(top_grad_h, top_grad_c)

    eval(test1.lstm.weight_hh.grad , sum(LSTM.grad_weight_hh))
    eval(test1.lstm.weight_ih.grad , sum(LSTM.grad_weight_ih))
    eval(test1.lstm.bias_ih.grad , sum(LSTM.grad_bias_ih))
    eval(test1.lstm.bias_hh.grad , sum(LSTM.grad_bias_hh))




    '''
    print("begin ------------conv---------------")
    print("---many element test -----")
    

    conv_1 = Conv2d(32, 32, 3, stride=2, padding=1)
    Convtest = torch.nn.Conv2d(32,32, 3, stride=2, padding=1)
    conv_1.weight = Convtest.weight.data
    conv_1.bias = Convtest.bias.data

    loss = 0
    top_grad_conv = []
    for i in range(1000):
        #conv_test
        input = torch.randn(32,6,6).unsqueeze(0)

        #forward_test
        result = Convtest(input)
        loss = loss + result
        my_result = conv_1.forward(input)
        top_grad_conv.append(torch.ones(result.shape))
        #backward_test
    loss.sum().backward()

    bottom_grad = conv_1.backward(top_grad_conv)
    eval(Convtest.weight.grad,sum(conv_1.grad_weight))
    eval(Convtest.bias.grad, sum(conv_1.grad_bias))


    print("begin ------------linear---------------")
    print("---many element test -----")
    #linear_test
    linear = torch.nn.Linear(256, 18)
    my_linear = Linear(256,18)
    my_linear.weight = linear.weight.data 
    my_linear.bias = linear.bias.data 

    loss = 0
    top_grad_linear = []
    for i in range(1000):
        input = torch.randn(1,256)
        weight = torch.randn(18,256)

        #forward_test
        result = linear(input)
        my_result = my_linear.forward(input)
        loss = loss +result.sum()
        #backward_test
        top_grad_linear.append(torch.ones(result.shape))
    
    loss.backward()
    bottom_grad = my_linear.backward(top_grad_linear)
    eval(linear.weight.grad, sum(my_linear.grad_weight))
    eval(linear.bias.grad, sum(my_linear.grad_bias))
    '''
    
    
    

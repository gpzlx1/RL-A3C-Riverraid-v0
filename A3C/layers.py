import torch 
import numpy as np

class Layer(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError 

    def load_weights(self):
        raise NotImplementedError 



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
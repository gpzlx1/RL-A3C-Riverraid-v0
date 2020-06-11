#	Technical report

[TOC]

## layer design

### LSTM

在使用深度学习处理时序问题时，RNN时最常使用的模型之一。RNN能够有效的将之前的时间片信息用于计算当前时间片的输入。其中Long Short Term Memory (LSTM)是一种常见且有效的神经网络。由于`Riverraid-v0`虽然`action`是离散的，但是其状态在时间尺度上有非常强的相关性，所以考虑使用LSTM进行训练，能取得不错的成果。



<img src="https://miro.medium.com/max/1631/0*tOgVu5w22Jg1yerG.png" alt="img" style="zoom:67%;" />

假设对`t`轮，对LSTM输入为$\mathbf{x}_t$，$\mathbf{h}_{t-1}$、$\mathbf{c}_{t-1}$，下面我们考虑$Forward$和$Backward$

* Forward

  LSTM内部由四个门构成，涉及到五个运算，分别是向量元素乘、向量和、$tanh$、$\sigma$ 以及四个门电路	

  对	$tanh(\mathbf{x}) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1} = \frac{1 - e^{-2x}}{1+ e^{-2x}}$, 对$\tanh$计算同样存在上溢或者下溢的问题。因此对正数，我们倾向于使用$\frac{1 - e^{-2x}}{1+ e^{-2x}}$计算；对于负数，我们倾向于使用$\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$以提高计算精度

  ```python
  #伪代码
  if x > 0:
      tanh = (1 - exp(-2 * x)) / (1 +  exp(-2 * x))
  else:
      tanh = (exp(2 * x) - 1) / (exp(2 * x) + 1)
      
  #实际代码
  def tanh(value):
      value = value.double()
      e_p = torch.exp(value.mul(2))
      e_n = torch.exp(value.mul(-2))
      tanh_n = (e_p - 1) / (e_p + 1)
      tanh_p = (1 - e_n) / (1 + e_n)
      return torch.where(value > 0, tanh_p, tanh_n).float()
  ```

  对 $\sigma(x) = \frac{e^{x}}{1 + e^{x}}$同样也存在这个问题，因而我们要区分正负数，进行单独计算，以提高精度

  ```python
  def sigmoid(value):
      value = value.double()
      sigmoid_value_p = torch.exp(-value).add(1).pow(-1)
      exp_value = torch.exp(value)
      sigmoid_value_n = exp_value.div(exp_value.add(1))
      return torch.where(value > 0, sigmoid_value_p, sigmoid_value_n).float()
  ```

  完成上诉设计，即可完成LSTM $Forward$设计，四个门

  ![img](https://miro.medium.com/max/620/1*Bqk-Ejg2WQzzngwKwiYvSw.gif)

  因此对当前LSTM最终输出为：

  ![img](https://miro.medium.com/max/470/1*bCG_X5bBbxr6_lE4dppZXg.gif)

  ```python
  def forward(input, hidden):
      hx, cx = hidden
      gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
      ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
      ingate     = F.sigmoid(ingate)
      forgetgate = F.sigmoid(forgetgate)
      cellgate   = F.tanh(cellgate)
      outgate    = F.sigmoid(outgate)
      cy = (forgetgate * cx) + (ingate * cellgate)
      hy = outgate * F.tanh(cy)
      return hy, cy
  ```

* Backward

  对LSTM backward相对而言就要非常复杂了，首先我们先对$\sigma$和$tanh$两个函数完成其对应的求导

  $\frac{\partial\sigma(x)}{\partial x} = \sigma(x) (1 - \sigma(x))$

  $\frac{\partial tanh{(x)}}{\partial x} = 1 - tanh^2{(x)}$

  根据LSTM前向传播计算LSTM的方向传播:
  
  ![img](https://miro.medium.com/max/742/1*cWEZJfk8ikLWj4xUS9T64w.gif)
  
  最终我们更新的权重为：
  
  ![img](https://miro.medium.com/max/322/1*DD_ocSrJ1Tvg6G5-8fft4Q.gif)
  
  实现部分非常长，这里仅放链接 [LSTM backward](https://github.com/gpzlx1/ML/blob/master/A3C/layers.py#L292)

### CONV

### Linear

# Reference

https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9


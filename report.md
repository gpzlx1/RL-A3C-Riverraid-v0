#	Technical report

[TOC]

## layer design

在本实验中，使用了四种网络结构:

**Conv2d**、**Linear**、**LSTM**、**elu**

### Conv2d

* forward 原理 & 实现

  直接将输入图像与卷积核进行[二维卷积](https://baike.baidu.com/item/卷积神经网络)

  利用[conv2d](https://pytorch.org/docs/stable/nn.functional.html#conv2d)来实现卷积运算：

  ```
  F.conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
  ```

* backward 原理

  用一个[简单的例子](http://staff.ustc.edu.cn/~jwangx/classes/210709/notes/Lec10.pdf)说明其原理（暂不考虑stride，下文补充）:

  ![例子](figures/1.jpg)

  ***首先求 $E$ 对权重 $F$ 的梯度***：
  $$
  \frac{\partial E}{\partial F_{11}} = \frac{\partial E}{\partial \mathbf{O}}^{\top}\frac{\partial \mathbf{O}}{\partial F_{11}}​= \frac{\partial E}{\partial O_{11}}X_{11}+\frac{\partial E}{\partial O_{12}}X_{12}+\frac{\partial E}{\partial O_{21}}X_{21}+\frac{\partial E}{\partial O_{22}}X_{22}
  $$

  以此类推，可以看出， $E$ 对权重 $F$ 的梯度也可以通过卷积运算求出

  ![](figures/2.jpg)

  ***同理，考虑  $E$ 对输入 $X$ 的梯度***：
  $$
  \frac{\partial E}{\partial X_{11}} = \frac{\partial E}{\partial \mathbf{O}}^{\top}\frac{\partial \mathbf{O}}{\partial X_{11}}​= \frac{\partial E}{\partial O_{11}}F_{11}+\frac{\partial E}{\partial O_{12}}0+\frac{\partial E}{\partial O_{21}}0+\frac{\partial E}{\partial O_{22}}0
  $$
  以此类推，我们发现可以将卷积核（权重$F$）旋转180°，并与 $\frac{\partial E}{\partial \mathbf{O}}$ （补0）进行卷积从而得到结果
  ![](figures/3.JPG)

  

  ***而当$stride>1$ 时***，需要将原本的卷积操作换为[空洞卷积](https://www.jianshu.com/p/f743bd9041b3)，dilation 的值即为 stride（将卷积核设为为$1\times1$即可证明）

  ![](figures/4.jpg)



* backward 实现

  记$O$为卷积结果，$W$为权重（卷积核），$X$为输入，$b$为偏置

  ***权重的梯度：***通过[F.conv2d](https://pytorch.org/docs/stable/nn.functional.html#conv2d)来实现卷积运算（设置空洞dilation）

  ***输入的梯度：***通过反卷积函数[conv_transpose2d](https://pytorch.org/docs/stable/nn.functional.html#conv-transpose2d)实现（反卷积的理解参考[这里](https://www.zhihu.com/question/48279880)，原理上文已描述过）

  ```
  conv_for_back = F.conv_transpose2d(top_grad_t,self.weight,torch.zeros(self.in_channels),self.stride,self.padding,				(self.input[i].shape[2]-top_grad.shape[2])%2,self.groups,self.dilation)
  ```

  ***偏置的梯度：***直接求和即可。
$$
  \frac{\partial Loss}{\partial b} = \sum \frac{\partial Loss}{O_{ij}}
$$

### Linear

* forward 原理 & 实现

  $$
  \mathbf{y} = \mathbf{x}W^{\top}+b
  $$
  利用[F.linear](https://pytorch.org/docs/stable/nn.functional.html#linear)函数即可

* backward 原理 & 实现

  ***权重的梯度：***
  $$
  \frac{\partial Loss}{\partial W} = \frac{\partial Loss}{\partial \mathbf{y}}^{\top}\mathbf{x}
  $$
  ***偏置的梯度：***
  $$
  \frac{\partial Loss}{\partial \mathbf{b}} = \frac{\partial Loss}{\partial \mathbf{y}}
  $$
  ***输入的梯度：***
  $$
  \frac{\partial Loss}{\partial \mathbf{x}} = \frac{\partial Loss}{\partial \mathbf{y}}\mathbf{W}
  $$
  以上梯度只需通过$torch.matmul$, $torch.add$函数即可求出。

  

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

  完成上诉设计，即可完成LSTM $Forward$设计，四个门:

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
  
  实现部分非常长，这里仅放实现代码的链接 [LSTM backward](https://github.com/gpzlx1/ML/blob/master/A3C/layers.py#L292)

## elu

激活函数elu非常简单

* Forward:
  $$
  f(x) = \begin{cases} x, &\text{if } x >0; \\  \alpha(\exp{(x)} - 1) & \text{if } x \leq 0\end{cases}
  $$

* Backward:
  $$
  \frac{\partial f(x)}{\partial x} = \begin{cases} 1, &\text{if } x >0; \\  \alpha\exp{(x)}& \text{if } x \leq 0\end{cases}
  $$
  在本实验，使用 $\alpha = 1$.

# Reference

https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9


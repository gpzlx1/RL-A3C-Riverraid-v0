# RL-A3C-Riverraid-v0
A simple implementation of A3C for atari game `Riverraid-v0` from scratch. 
* Only use torch data structure and some commonly used torch functions.
* implement my own Conv2d, Linear and LSTM layers, including forward and backward.
## How to train
```shell
  virtualenv --python=python3 venv
  source venv/bin/activate
  
  # install dependencies
  pip install -r requirement.txt
  
  # train
  python A3C/my_main.py
```
The detail hyperparameters are configed by python class `Config` in `my_main.py`.
Train model and log would be stored in `./model` and `./log` separately.
## How to test
No implementation. By you can implement your test file refer to [PB17111656.py](https://github.com/gpzlx1/ML/blob/master/PB17111656.py).
## Result
![100](https://github.com/gpzlx1/ML/blob/master/figures/learning_curve_plot.png)


![gif](https://github.com/gpzlx1/ML/blob/master/figures/result.gif)

## Reference

[pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)

[torch.optim.adam](https://github.com/pytorch/pytorch/blob/6e2bb1c05442010aff90b413e21fce99f0393727/torch/optim/adam.py)

[Welcome to Deep Reinforcement Learning Part 1 : DQN](https://towardsdatascience.com/welcome-to-deep-reinforcement-learning-part-1-dqn-c3cab4d41b6b)

[百度百科：卷积神经网络](https://baike.baidu.com/item/卷积神经网络)

[pytorch官方文档：nn.functional](https://pytorch.org/docs/stable/nn.functional.html)

[机器学习课件/Lec10.pdf](http://staff.ustc.edu.cn/~jwangx/classes/210709/notes/Lec10.pdf)

[空洞卷积理解](https://www.jianshu.com/p/f743bd9041b3)

[怎样通俗易懂地解释反卷积？](https://www.zhihu.com/question/48279880)

[Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf)

[AC、A2C、A3C算法](https://zhuanlan.zhihu.com/p/62100741)

[Actor-Critic](https://www.cnblogs.com/pinard/p/10272023.html)

[深度强化学习算法 A3C](https://www.cnblogs.com/wangxiaocvpr/p/8110120.html)

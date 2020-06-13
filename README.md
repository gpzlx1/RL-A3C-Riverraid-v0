# ML
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
## How to test
No implementation. By you can implement your test file refer to [PB17111656.py](https://github.com/gpzlx1/ML/blob/master/PB17111656.py).
## Result
![100](https://github.com/gpzlx1/ML/blob/master/figures/learning_curve_plot.png)


![gif](https://github.com/gpzlx1/ML/blob/master/figures/result.gif)

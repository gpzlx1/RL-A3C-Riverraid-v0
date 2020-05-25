from copy import deepcopy
from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file
import torch.nn.functional as F

#!!!
from torch import from_numpy
#!!!

from torch import unsqueeze
import numpy as np
import math

def AtariRescale42x42(src, new_size):
    src = src.astype("float32")
    H, W, C = src.shape
    print(H,W)
    new_H, new_W = new_size
    dst = np.zeros((new_H, new_W, C))
    ratio_h = H / new_H
    ratio_w = W / new_W
    for i in range(new_H):
        src_y = ratio_h * i
        y = math.floor(src_y)
        v = src_y - y
        if y < 0:
            y = 0
            v = 0
        elif y >= H - 1:
            y = H - 2
            v = 1

        for j in range(new_W):
            src_x = ratio_w * j
            x = math.floor(src_x)
            u = src_x - x
            if x < 0:
                x = 0
                u = 0
            elif x >= W - 1:
                x = W - 2
                u = 1

            dst[i][j] = (1-v) * (1-u) *  src[y][x] +  v * (1-u) * src[y+1][x] \
                        + (1-v) * u * src[y][x+1] +  v * u * src[y+1][x+1]

    return dst.astype("uint8")


class NormalizedEnv():
    def __init__(self):
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)

def conv(input,params):
    pass
def linear(input,params):
    pass

class PB17111656(RL_alg):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)

        self.team = ['PB17111656','PB17030808']  # 记录队员学号
        self.config = get_params_from_file('src.alg.PB17111656.rl_configs', params_name='params')  # 传入参数

        self.ac_space = ac_space
        self.state_dim = ob_space.shape[0]
        self.action_dim = ac_space.n
        self.normalize = NormalizedEnv()

    def step(self, state):
        '''
        state = AtariRescale42x42(state,[42,42])
        state = self.normalize.observation(state)
        state = from_numpy(state)#需手动实现
        params = get_params_from_file('src.alg.PB17111656.param')#将存储在当前目录下训练好的参数读入
        x = conv(unsqueeze(state,0),params)
        logit = linear(x,params)
        #不清楚非训练过程需不需要lstm
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()[0,0]
        '''
        action = self.ac_space.sample()
        return action

    def explore(self, obs):
        # currently, we only need to implement explore function
        raise NotImplementedError

    def test(self):
        print('ok1')

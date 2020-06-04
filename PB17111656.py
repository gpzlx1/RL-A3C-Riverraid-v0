import os
import torch
from gym.spaces import Discrete
from gym.spaces.box import Box
import torch.nn.functional as F
from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file
import numpy as np
import math
import cv2

import sys
sys.path.append('src/alg/PB17111656/A3C')
from my_model import AcotrCritic

def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame

class NormalizedEnv():
    def __init__(self):
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def observation(self, observation):
        observation = _process_frame42(observation)
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


def find_model(filename):
    list_models = os.listdir(filename)
    x = ''
    max = 0
    for model in list_models:
        num = int(model[model.find('_')+1:])
        if num > max:
            max = num
            x = model
    return filename+'/'+x



class PB17111656(RL_alg):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)

        self.team = ['PB17111656','PB17030808']  # 记录队员学号
        self.config = get_params_from_file('src.alg.PB17111656.rl_configs', params_name='params')  # 传入参数

        self.ac_space = ac_space
        self.state_dim = ob_space.shape[0]
        self.action_dim = ac_space.n
        self.action_dim = ac_space.n
        self.normalize = NormalizedEnv()
        self.model = AcotrCritic(self.state_dim, self.ac_space, model_train=False)
        self.hx = torch.zeros(1, 256)
        self.cx = torch.zeros(1, 256)
        #读取model文件夹的内容
        params = find_model('src/alg/PB17111656/model')
        self.model.load_model(params)

    def step(self, state):
        state = self.normalize.observation(state)
        state = torch.Tensor(state)
        _, logit, (self.hx, self.cx) = self.model.forward((state.unsqueeze(0), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        return action[0,0]

    def reset(self):
        self.hx = torch.zeros(1, 256)
        self.cx = torch.zeros(1, 256)


    def explore(self, obs):
        # how to train
        # cd PB17111656
        # python3 A3C/my_main.py
        raise NotImplementedError

    def test(self):
        print('ok1')


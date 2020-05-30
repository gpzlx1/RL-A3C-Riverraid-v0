import gym
import numpy as np
import math
import cv2
from gym.spaces.box import Box




def resize(src, new_size):
    src = src.astype("float32")
    H, W, C = src.shape
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
    

# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    return env


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


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def observation(self, observation):
        return _process_frame42(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
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


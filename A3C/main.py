from __future__ import print_function

import os

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train

class config(object):
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.gae_lambda = 1.00
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.01
        self.max_grad_norm = 50
        self.seed = 1
        self.num_processes = 4
        self.num_steps = 20
        self.max_episode_length = 100
        self.env_name = 'Riverraid-v0'
        self.no_shared = False


if __name__ == '__main__':
    params = config()
    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""


    torch.manual_seed(params.seed)
    env = create_atari_env(params.env_name)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if params.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(params.num_processes, params, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(0, params.num_processes):
        p = mp.Process(target=train, args=(rank, params, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

import os 

import torch
import torch.multiprocessing as mp 

import my_optim
from envs import create_atari_env
from my_model import AcotrCritic
from my_test import test
from my_train import train

class config(object):
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.gae_lambda = 1.00
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 50
        self.seed = 1
        self.num_processes = 23
        self.num_steps = 20
        self.max_step_length = 1000000
        self.env_name = 'Riverraid-v0'
        self.model_path = './model/'
        self.test_interval = 20

if __name__ == '__main__':
    #env
    args = config()
    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""


    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    shared_model = AcotrCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter, "./log/"))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
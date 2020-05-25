import os
from importlib import import_module
import gym

#为了在windows上临时测试代码，可删除
import sys
sys.path.insert(0, os.getcwd())

from src.utils.misc_utils import get_params_from_file
from src.utils.pipeline import evaluate
from src.alg.RL_alg import RL_alg

def init_env_policy(alg, env_name):
    env = gym.make(env_name)
    print('Env:',env_name)
    
    policy = alg(env.observation_space,env.action_space)
    print('Load alg')

    return env, policy

if __name__ == '__main__':
    main_params = get_params_from_file('configs.main_setting',params_name='params')

    list_modules=os.listdir('../src/alg')
    list_modules.remove('RL_alg.py')
    for module_name in list_modules:
        if 'PB' in module_name:
            print("Load module ",module_name)
            getattr(import_module('.'.join(['src', 'alg', module_name,module_name])),module_name)
    if os.path.exists('./score.csv'):
        os.remove('./score.csv')
    for alg in RL_alg.__subclasses__():
        env, policy = init_env_policy(alg, main_params['env_name'])
        evaluate(env,policy,**main_params['evaluate'])
        env.close()
        del env, policy
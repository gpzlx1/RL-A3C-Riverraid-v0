from copy import deepcopy
from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file


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
        action = self.ac_space.sample()
        return action

    def explore(self, obs):
        # currently, we only need to implement explore function
        raise NotImplementedError

    def test(self):
        print('ok1')

from envs import create_atari_env
from model import ActorCritic
import my_model

env = create_atari_env("Riverraid-v0")

model = ActorCritic(env.observation_space.shape[0], env.action_space)
m_model = my_model.AcotrCritic(env.observation_space.shape[0], env.action_space)

for i in model.parameters():
    print(i.shape)

print()
for i in m_model.parameters():
    print(i.shape)
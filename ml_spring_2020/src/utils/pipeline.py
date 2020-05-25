import os
import gym


def evaluate(env: object, policy: object, num_evaluate_episodes: object, is_render: object) -> object:
    for j in range(num_evaluate_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0
        ep_len = 0
        while not(done):
            if is_render:
                env.render()
            # Take deterministic actions at test time 
            ac = policy.step(obs)
            obs, reward, done, _ = env.step(ac)
            ep_ret += reward
            ep_len += 1
        policy.logkv_mean("TestEpRet", ep_ret)
        policy.logkv_mean("TestEpLen", ep_len)
    policy.dumpkvs()
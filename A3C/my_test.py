import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from my_model import AcotrCritic


def test(rank, args, shared_model, counter, log_path):

    log_file = open(log_path + time.asctime(time.localtime()), "w")

    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = AcotrCritic(env.observation_space.shape[0], env.action_space)

    state = env.reset()
    state = torch.Tensor(state)
    rewards = []
    reward_sum = 0
    max_100_episode_reward = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=300)
    step_length = 0
    try:

        while True:
            step_length += 1
            #sync
            if done:
                model.get_parameters(shared_model.parameters())
                model.clear_temp()
                cx = torch.zeros(1, 256)
                hx = torch.zeros(1, 256)


            value, logit, (hx, cx) = model.forward((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1, keepdim=True)[1].numpy()

            state, reward, done, _ = env.step(action[0, 0])
            done = done or step_length >= args.max_step_length
            reward_sum += reward

            # a quick hack to prevent the agent from stucking
            actions.append(action[0, 0])
            if actions.count(action[0]) == actions.maxlen:
                done = True

            if done:
                rewards.append(reward_sum)
                mean_100_episode = sum(rewards[-100:]) / len(rewards[-100:])
                mess = "Time: {}, num steps {}, FPS {:.0f}, current reward {}, step length {}, 100-episode reward {}\n".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    counter.value, 
                    counter.value / (time.time() - start_time),
                    reward_sum,
                    step_length,
                    mean_100_episode
                )
                log_file.write(mess)
                reward_sum = 0
                step_length = 0
                actions.clear()
                state = env.reset()
                time.sleep(45)

                if mean_100_episode > max_100_episode_reward:
                    max_100_episode_reward = mean_100_episode
                    model.save_model(args.model_path + "checkpoint_" + str(max_100_episode_reward))

            state = torch.Tensor(state)

    except KeyboardInterrupt:
        log_file.write("finish\n")
        log_file.close()
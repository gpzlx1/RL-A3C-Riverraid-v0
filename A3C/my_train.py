import torch
import torch.nn.functional as F

from envs import create_atari_env
from my_model import AcotrCritic


def clip_grad(parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        return total_norm


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def grad_loss(values, logits, rewards, actions, params,R):
    grad_value = []
    gae = torch.zeros(1, 1)
    grad_logits = []
    for i in reversed(range(len(rewards))):
        # grad_value
        R = params.gamma * R + rewards[i]
        grad_value.append(params.value_loss_coef * (values[i] - R))  # value累和

        # grad_logit
        delta_t = rewards[i] + params.gamma * \
                  values[i + 1] - values[i]
        gae = gae * params.gamma * params.gae_lambda + delta_t
        grad_log_probs = -gae.detach()
        grad_entropies = params.entropy_coef
        prob = F.softmax(logits[i], dim=-1)
        log_prob = F.log_softmax(logits[i], dim=-1)
        grad_logits_log = torch.zeros(logits[i].shape)
        grad_logits_ent = torch.zeros(logits[i].shape)
        # grad_entropies
        for j in range(logits[i].shape[1]):
            for k in range(logits[i].shape[1]):
                if k == j:
                    grad_logits_ent[0][j] += (1 + log_prob[0][k]) * prob[0][j] * (1 - prob[0][j])
                else:
                    grad_logits_ent[0][j] += (1 + log_prob[0][k]) * prob[0][k] * (-prob[0][j])
        grad_logits_ent = torch.mul(grad_entropies, grad_logits_ent)
        # grad_log_prob
        for j in range(logits[i].shape[1]):
            if actions[i] == j:
                grad_logits_log[0][j] += grad_log_probs[0][0] * (1 - prob[0][j])
            else:
                grad_logits_log[0][j] += -grad_log_probs[0][0] * prob[0][j]
        grad_logits.append(torch.add(grad_logits_log, grad_logits_ent))
    return grad_value[::-1], grad_logits[::-1]


def train(rank, args, shared_model, counter, lock, optimizer):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = AcotrCritic(env.observation_space.shape[0], env.action_space)

    model.clear_temp()

    state = env.reset()
    state = torch.Tensor(state)
    done = True

    step_length = 0
    while True:
        #sync 
        model.get_parameters(shared_model.parameters())
        model.clear_temp()
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        logits = []
        actions = []

        for step in range(args.num_steps):
            step_length += 1
            value, logit, (hx, cx) = model.forward((state.unsqueeze(0), (hx, cx)))

            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            action = prob.multinomial(num_samples=1)
            
            state, reward, done, _ = env.step(action.numpy())

            done = done or step_length >= args.max_step_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                step_length = 0
                state = env.reset()

            state = torch.Tensor(state)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            logits.append(logit)
            actions.append(action)

            if done:
                break
        

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model.forward((state.unsqueeze(0), (hx, cx)))
            R = value
            
        values.append(R)
        gae = torch.zeros(1, 1)

        #compute loss and grad for value and logit
        top_grad_value, top_grad_logit = grad_loss(values, logits, 
                rewards, actions, args, R)

        #update weight
        optimizer.zero_grad()
        model.backward(top_grad_value. top_grad_logit)
        ensure_shared_grads(model, shared_model)
        optimizer.step()

        

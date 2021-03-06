import torch
import torch.nn.functional as F

from envs import create_atari_env
from my_model import AcotrCritic

def norm2(tensor):
    tensor = tensor.double()
    tensor = tensor.pow(2)
    return torch.sqrt(torch.sum(tensor)).float()

def clip_grad(parameters, max_norm):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        max_norm = float(max_norm)
        total_norm = norm2(torch.stack([norm2(p.grad) for p in parameters]))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.mul_(clip_coef)
        return total_norm


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def grad_loss(values, logits, rewards, actions, params, R):
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
        grad_log_probs = -gae
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

            #print(logit)
            prob = F.softmax(logit, dim=-1)
            #print(prob)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1)
            actions.append(action)
            log_prob = log_prob.gather(1, action)

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
            logits.append(logit)
            

            if done:
                break
        

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model.forward((state.unsqueeze(0), (hx, cx)))
            R = value
            
        values.append(R)

        optimizer.zero_grad()
        #compute loss and grad for value and logit
        top_grad_value, top_grad_logit = grad_loss(values, logits, 
                rewards, actions, args, R)

        #update weight
        model.backward(top_grad_value, top_grad_logit)
        clip_grad(model.parameters(), args.max_grad_norm)
        ensure_shared_grads(model, shared_model)
        optimizer.step()

        

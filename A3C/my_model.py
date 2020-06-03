import numpy as np
import torch
import torch.nn.functional as F
import layers


def grad_elu(grad_y, x):
    # assume for elu, alpha = 1
    bottom_grad_list = []
    zero = torch.zeros(x[0].shape)
    for i in range(len(grad_y)):
        bottom_grad = torch.where(x[i] > 0, zero, x[i])
        bottom_grad_list.append(bottom_grad.add(1).mul(grad_y[i]))
    return bottom_grad_list


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



class AcotrCritic(object):
    def __init__(self, num_inputs, action_space):
        self.conv1 = layers.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = layers.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = layers.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = layers.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = layers.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space.n
        self.critic_linear = layers.Linear(256, 1)
        self.actor_linear = layers.Linear(256, num_outputs)

        # initial paramater
        self.conv1.init_weight(random=True)
        self.conv1.init_bias(random=False)
        self.conv2.init_weight(random=True)
        self.conv2.init_bias(random=False)
        self.conv3.init_weight(random=True)
        self.conv3.init_bias(random=False)
        self.conv4.init_weight(random=True)
        self.conv4.init_bias(random=False)

        self.critic_linear.init_weight(random=True)
        self.critic_linear.init_bias(random=False)
        self.actor_linear.init_weight(random=True)
        self.actor_linear.init_bias(random=False)

        self.lstm.init_weight(random=True)
        self.lstm.init_bias(random=False)

        # grad
        self.y1 = []
        self.y2 = []
        self.y3 = []
        self.y4 = []

    def clear_temp(self):
        self.y1.clear()
        self.y2.clear()
        self.y3.clear()
        self.y4.clear()

        self.conv1.clear_temp()
        self.conv2.clear_temp()
        self.conv3.clear_temp()
        self.conv4.clear_temp()

        self.lstm.clear_temp()

        self.actor_linear.clear_temp()
        self.critic_linear.clear_temp()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs

        x = F.elu(self.conv1.forward(inputs))
        self.y1.append(x)

        x = F.elu(self.conv2.forward(x))
        self.y2.append(x)

        x = F.elu(self.conv3.forward(x))
        self.y3.append(x)

        x = F.elu(self.conv4.forward(x))
        self.y4.append(x)

        # x.shape = 1, 32, 3, 3
        x = x.view(-1, 32 * 3 * 3)
        # x.shape = 1, 288

        hx, cx = self.lstm.forward(x, (hx, cx))
        x = hx

        return self.critic_linear.forward(x), self.actor_linear.forward(x), (hx, cx)

    def backward(self, top_grad_value, top_grad_logit):
        grad_inputs = []
        if len(top_grad_logit) != len(top_grad_logit):
            print("error in model backward")
            raise NotImplementedError

        grad_critic_linear = self.critic_linear.backward(top_grad_value)
        grad_actor_liner = self.actor_linear.backward(top_grad_logit)

        top_grad_h = []

        for i in range(len(grad_critic_linear)):
            top_grad_h.append(grad_critic_linear[i] + grad_actor_liner[i])

        top_grad_c = [0] * len(grad_critic_linear)

        top_grad_conv4, _, _ = self.lstm.backward(top_grad_h, top_grad_c)

        top_grad_conv4 = [element.view(-1, 32, 3, 3) for element in top_grad_conv4]

        top_grad_conv4 = grad_elu(top_grad_conv4, self.y4)
        top_grad_conv3 = self.conv4.backward(top_grad_conv4)

        top_grad_conv3 = grad_elu(top_grad_conv3, self.y3)
        top_grad_conv2 = self.conv3.backward(top_grad_conv3)

        top_grad_conv2 = grad_elu(top_grad_conv2, self.y2)
        top_grad_conv1 = self.conv2.backward(top_grad_conv2)

        top_grad_conv1 = grad_elu(top_grad_conv1, self.y1)
        grad_inputs = self.conv1.backward(top_grad_conv1)

        return grad_inputs

    def parameters(self):
        yield self.conv1.weight
        yield self.conv1.bias

        yield self.conv2.weight
        yield self.conv2.bias

        yield self.conv3.weight
        yield self.conv3.bias

        yield self.conv4.weight
        yield self.conv4.bias

        yield self.lstm.weight_ih
        yield self.lstm.weight_hh
        yield self.lstm.bias_ih
        yield self.lstm.bias_hh

        yield self.critic_linear.weight
        yield self.critic_linear.bias

        yield self.actor_linear.weight
        yield self.actor_linear.bias

    def share_memory(self):
        for i in self.parameters():
            i.data.share_memory_()

    def clip_grad(self,parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        return total_norm

    def save_model(self,filename = 'trained_parameter'):
        parameters = [p for p in self.parameters()]
        torch.save(parameters,'./'+filename)

    def load_model(self,filename = 'trained_parameter'):
        parameters = torch.load('./'+filename)
        self.get_parameters(parameters)

    def get_parameters(self, parameters):
        for dst, src in zip(self.parameters(), parameters):
            dst.data =  src.data.clone()






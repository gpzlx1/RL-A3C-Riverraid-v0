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

def grad_loss(values,logits,rewards,actions,params):
    R = torch.zeros(1, 1)
    grad_value = []
    gae = torch.zeros(1, 1)
    grad_logits = []
    for i in reversed(range(len(rewards))):
        #grad_value
        R = params.gamma * R + rewards[i]
        grad_value.append(params.value_loss_coef*(values[i] - R))    #value累和

        #grad_logit
        delta_t = rewards[i] + params.gamma * \
                  values[i + 1] - values[i]
        gae = gae * params.gamma * params.gae_lambda + delta_t
        grad_log_probs = -gae.detach()
        grad_entropies = params.entropy_coef
        prob = F.softmax(logits[i], dim=-1)
        log_prob = F.log_softmax(logits[i], dim=-1)
        grad_logits_log = torch.zeros(logits[i].shape)
        grad_logits_ent = torch.zeros(logits[i].shape)
        #grad_entropies
        for j in range(logits[i].shape[1]):
            for k in range(logits[i].shape[1]):
                if k == j:
                    grad_logits_ent[0][j] +=  (1 + log_prob[0][k]) * prob[0][j]*(1-prob[0][j])
                else:
                    grad_logits_ent[0][j] +=  (1 + log_prob[0][k]) * prob[0][k]*(-prob[0][j])
        grad_logits_ent = torch.mul(grad_entropies,grad_logits_ent)
        # grad_log_prob
        for j in range(logits[i].shape[1]):
            if actions[i] == j:
                grad_logits_log[0][j] += grad_log_probs[0][0]*(1-prob[0][j])
            else:
                grad_logits_log[0][j] += -grad_log_probs[0][0]*prob[0][j]
        grad_logits.append(torch.add(grad_logits_log,grad_logits_ent))
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

        #initial paramater
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
        
        #grad
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.x4 = []

    def clear_grad(self):
        self.x1.clear()
        self.x2.clear()
        self.x3.clear()
        self.x4.clear()
        
        self.conv1.clear_grad()
        self.conv2.clear_grad()
        self.conv3.clear_grad()
        self.conv4.clear_grad()

        self.lstm.clear_grad()
        
        self.actor_linear.clear_grad()
        self.critic_linear.clear_grad()



    def forward(self, inputs):
        inputs, (hx, cx) = inputs

        x = self.conv1.forward(inputs)
        self.x1.append(x)
        x = F.elu(x)
        
        x = self.conv2.forward(x)
        self.x2.append(x)
        x = F.elu(x)

        x = self.conv3.forward(x)
        self.x3.append(x)
        x = F.elu(x)

        x = self.conv4.forward(x)
        self.x4.append(x)
        x = F.elu(x)

        # x.shape = 1, 32, 3, 3
        x = x.view(-1, 32 * 3 * 3)
        # x.shape = 1, 288

        hx, cx = self.lstm.forward(x, (hx,cx))
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
        
        for  i in range(len(grad_critic_linear)):
            top_grad_h.append(grad_critic_linear[i] + grad_actor_liner[i])
        
        top_grad_c = [0] * len(grad_critic_linear)

        top_grad_conv4, _, _ = self.lstm.backward(top_grad_h, top_grad_c)

        top_grad_conv4 = [ element.view(-1, 32, 3, 3) for element in top_grad_conv4 ]

        top_grad_conv4 = grad_elu(top_grad_conv4, self.x4)
        top_grad_conv3 = self.conv4.backward(top_grad_conv4)

        top_grad_conv3 = grad_elu(top_grad_conv3, self.x3)
        top_grad_conv2 = self.conv3.backward(top_grad_conv3)

        top_grad_conv2 = grad_elu(top_grad_conv2, self.x2)
        top_grad_conv1 = self.conv2.backward(top_grad_conv2)

        top_grad_conv1 = grad_elu(top_grad_conv1, self.x1)
        grad_inputs = self.conv1.backward(top_grad_conv1)
        
        return grad_inputs



    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError



def eval(value1, value2):
    if value1.shape != value2.shape:
        print("error")
        return 
    
    print(torch.max(torch.abs((value1 + 1e-6)/(value2 + 1e-6))),torch.min(torch.abs((value1 + 1e-6)/(value2 + 1e-6))),torch.mean(torch.abs(torch.abs((value1 + 1e-6).div(value2 + 1e-6)) - 1)))




def copy_weight(model, my_model):
    print("-- conv --")
    if my_model.conv1.weight.shape == model.conv1.weight.shape:
        my_model.conv1.weight = model.conv1.weight.data
    else:
        print("error")

    if my_model.conv2.weight.shape == model.conv2.weight.shape:
            my_model.conv2.weight = model.conv2.weight.data
    else:
        print("error")

    if my_model.conv3.weight.shape == model.conv3.weight.shape:
            my_model.conv3.weight = model.conv3.weight.data
    else:
        print("error")
    
    if my_model.conv4.weight.shape == model.conv4.weight.shape:
            my_model.conv4.weight = model.conv4.weight.data
    else:
        print("error")



    if my_model.conv1.bias.shape == model.conv1.bias.shape:
            my_model.conv1.bias = model.conv1.bias.data
    else:
        print("error")

    if my_model.conv2.bias.shape == model.conv2.bias.shape:
            my_model.conv2.bias = model.conv2.bias.data
    else:
        print("error")
        
    if my_model.conv3.bias.shape == model.conv3.bias.shape:
            my_model.conv3.bias = model.conv3.bias.data
    else:
        print("error")

    if my_model.conv4.bias.shape == model.conv4.bias.shape:
            my_model.conv4.bias = model.conv4.bias.data
    else:
        print("error")

    print("-- linear --")

    if my_model.critic_linear.weight.shape == model.critic_linear.weight.shape:
        my_model.critic_linear.weight = model.critic_linear.weight.data
    else:
        print("error")

    if my_model.actor_linear.weight.shape == model.actor_linear.weight.shape:
        my_model.actor_linear.weight = model.actor_linear.weight.data
    else:
        print("error")

    if my_model.critic_linear.bias.shape == model.critic_linear.bias.shape:
        my_model.critic_linear.bias = model.critic_linear.bias.data
    else:
        print("error")

    if my_model.actor_linear.bias.shape == model.actor_linear.bias.shape:
        my_model.actor_linear.bias = model.actor_linear.bias.data
    else:
        print("error")
    
    print("-- lstm --")
    if my_model.lstm.weight_hh.shape == model.lstm.weight_hh.shape and my_model.lstm.weight_ih.shape == model.lstm.weight_ih.shape:
        my_model.lstm.weight_hh = model.lstm.weight_hh.data
        my_model.lstm.weight_ih = model.lstm.weight_ih.data
    else:
        print("error")

    if my_model.lstm.bias_ih.shape == model.lstm.bias_ih.shape and my_model.lstm.bias_hh.shape == model.lstm.bias_hh.shape:
        my_model.lstm.bias_ih = model.lstm.bias_ih.data
        my_model.lstm.bias_hh = model.lstm.bias_hh.data
    else:
        print("error")

def check(model, my_model):
    print("--- linear")
    eval(my_model.actor_linear.grad_weight, model.actor_linear.weight.grad)
    eval(my_model.actor_linear.grad_bias, model.actor_linear.bias.grad)
    eval(my_model.critic_linear.grad_weight, model.critic_linear.weight.grad)
    eval(my_model.critic_linear.grad_bias, model.critic_linear.bias.grad)

    print("--- lstm")
    eval(my_model.lstm.grad_bias_hh, model.lstm.bias_hh.grad)
    eval(my_model.lstm.grad_bias_ih, model.lstm.bias_ih.grad)
    eval(my_model.lstm.grad_weight_ih, model.lstm.weight_ih.grad)
    eval(my_model.lstm.grad_weight_hh, model.lstm.weight_hh.grad)

    print("--- conv")
    eval(my_model.conv4.grad_weight, model.conv4.weight.grad)
    eval(my_model.conv4.grad_bias, model.conv4.bias.grad)

    eval(my_model.conv3.grad_weight, model.conv3.weight.grad)
    eval(my_model.conv3.grad_bias, model.conv3.bias.grad)

    eval(my_model.conv2.grad_weight, model.conv2.weight.grad)
    eval(my_model.conv2.grad_bias, model.conv2.bias.grad)

    eval(my_model.conv1.grad_weight, model.conv1.weight.grad)
    eval(my_model.conv1.grad_bias, model.conv1.bias.grad)

    print("--- input")
    #eval(grad_inputs, inputs.grad)


def model_backward(model, my_model):
    top_grad_logit = []
    top_grad_value = []
    loss = 0
    my_model.clear_grad()
    for i in range(50):
        inputs = torch.randn(state.unsqueeze(0).shape)
        cx = torch.randn(1, 256)
        hx = torch.randn(1, 256)

        my_value, my_logit, _ = my_model.forward((inputs, (hx, cx)))
        value, logit, _ = model((inputs, (hx, cx)))

        loss =  value + logit.sum() + loss
        
        top_grad_logit.append(torch.ones(logit.shape))
        top_grad_value.append(torch.ones(value.shape))

    return top_grad_logit, top_grad_value, loss

if __name__ == "__main__":
    import random 
    from envs import create_atari_env
    import model
    print("-----------------test model forward-------------")
    print("-------------many element ---------------")
    env = create_atari_env("Riverraid-v0")
    my_model = AcotrCritic(env.observation_space.shape[0], env.action_space)
    my_model.clear_grad()

    model = model.ActorCritic(env.observation_space.shape[0], env.action_space)
    model.train()
    state = env.reset()
    state = torch.Tensor(state)

    copy_weight(model,my_model)

    
    print("---- checkout model backward ----")
    top_grad_logit, top_grad_value, loss = model_backward(model, my_model)
    loss.backward()
    my_model.backward(top_grad_value, top_grad_logit)
    check(model, my_model)

    '''

    print("---- checkout loss backward ----")
    values = []
    log_probs = []
    rewards = []
    entropies = []
    logits = []
    actions = []

    from main import config
    args = config()
    
    done = False
    cx = torch.zeros(1, 256)
    hx = torch.zeros(1, 256)
    my_hx = hx
    my_cx = cx
    episode_length = 0
    my_values = []
    my_logits = []
    for step in range(1):
        episode_length += 1
        my_value, my_logit, (my_hx, my_cx) = my_model.forward((state.unsqueeze(0),
                                            (my_hx, my_cx)))
        my_values.append(my_value)
        my_logits.append(my_logit)
        value, logit, (hx, cx) = model((state.unsqueeze(0),
                                            (hx, cx)))
 
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        entropies.append(entropy)

        action = prob.multinomial(num_samples=1).detach()
        actions.append(action)
        log_prob = log_prob.gather(1, action)
        
        state, reward, done, _ = env.step(action.numpy())
        
        reward = 20.0 if random.random() > 0.5 else 0.0
        print(reward)
        done = done or episode_length >= args.max_episode_length
        reward = max(min(reward, 1), -1)


        if done:
            episode_length = 0
            state = env.reset()
            done = False

        state = torch.from_numpy(state)
        values.append(value)
        log_probs.append(log_prob)
        rewards.append(reward)


        if done:
            break

    R = torch.zeros(1, 1)
    if not done:
        value, _, _ = model((state.unsqueeze(0), (hx, cx)))
        R = value.detach()
    

    values.append(R)
    my_values.append(R)
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1)
    for i in reversed(range(len(rewards))):
        R = args.gamma * R + rewards[i]
        advantage = R - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        # Generalized Advantage Estimation
        delta_t = rewards[i] + args.gamma * \
                  values[i + 1] - values[i]
        gae = gae * args.gamma * args.gae_lambda + delta_t

        policy_loss = policy_loss - \
                      log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

    (policy_loss + args.value_loss_coef * value_loss).backward()
    top_grad_value, top_grad_logit = grad_loss(my_values, my_logits, rewards, actions, args)
    my_model.backward(top_grad_value, top_grad_logit)
    check(model, my_model)
    temp = 0
    for i in range(len(values)):
        temp = temp + abs(values[i] - my_values[i])
    print(temp)

    temp = 0
    for i in range(len(values)):
        temp = temp + torch.max(torch.abs(values[i] - my_values[i]))
    print(temp)
    '''

import numpy as np 
import torch
import torch.nn.functional as F
import layers


def grad_elu(top_grad):
    # assume for elu, alpha = 1
    zero = torch.zeros(top_grad.shape)
    bottom_grad = torch.where(top_grad > 0, zero, top_grad)
    return bottom_grad + 1


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
        


    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1.forward(inputs))
        x = F.elu(self.conv2.forward(x))
        x = F.elu(self.conv3.forward(x))
        x = F.elu(self.conv4.forward(x))
        # x.shape = 1, 32, 3, 3
        x = x.view(-1, 32 * 3 * 3)
        # x.shape = 1, 288

        hx, cx = self.lstm.forward(x, (hx,cx))
        x = hx

        return self.critic_linear.forward(x), self.actor_linear.forward(x), (hx, cx)

    def backward(self, top_grad_value, top_grad_logit):
        grad_critic_linear = self.critic_linear.backward(top_grad_value)
        grad_actor_liner = self.actor_linear.backward(top_grad_logit)
        
        top_grad_lstm = grad_critic_linear + grad_actor_liner

        top_grad_conv4, _, _ = self.lstm.backward(top_grad_lstm, None)

        top_grad_conv4 = top_grad_conv4.view(-1, 32, 3, 3)

        top_grad_conv4 = grad_elu(top_grad_conv4)
        top_grad_conv3 = self.conv4.backward(top_grad_conv4)

        top_grad_conv3 = grad_elu(top_grad_conv3)
        top_grad_conv2 = self.conv3.backward(top_grad_conv3)

        top_grad_conv2 = grad_elu(top_grad_conv2)
        top_grad_conv1 = self.conv2.backward(top_grad_conv2)

        top_grad_conv1 = grad_elu(top_grad_conv1)
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
    
    print(torch.max(torch.abs((value1 - value2) / value2)))


if __name__ == "__main__":
    inputs = torch.ones(1,128) * 2
    inputs.requires_grad = True
    linear1 = torch.nn.Linear(128,1)
    linear2 = torch.nn.Linear(128,18)
    value = linear1(inputs)
    logit = linear2(inputs)
    loss = value + logit.sum()
    loss.backward()
    #print(linear1.weight)
    #print(inputs.grad)

    my_linear1 = layers.Linear(128,1)
    my_linear1.weight = linear1.weight.data
    my_linear1.bias = linear1.bias.data
    my_linear2 = layers.Linear(128,18)
    my_linear2.weight = linear2.weight.data
    my_linear2.bias = linear2.bias.data

    my_value = my_linear1.forward(inputs)
    my_logit = my_linear2.forward(inputs)
    my_loss = my_value + (my_logit * 0.5).sum()

    print(loss, my_loss)

    a = my_linear1.backward(torch.ones(my_value.shape))
    b = my_linear2.backward(torch.ones(my_logit.shape) * 0.7)
    eval(a+b, inputs.grad)
'''
if __name__ == "__main__":
    from envs import create_atari_env
    import model
    print("-----------------test model forward-------------")
    env = create_atari_env("Riverraid-v0")
    my_model = AcotrCritic(env.observation_space.shape[0], env.action_space)
    model = model.ActorCritic(env.observation_space.shape[0], env.action_space)
    state = env.reset()
    state = torch.Tensor(state)
    state.requires_grad = True
    cx = torch.zeros(1, 256)
    hx = torch.zeros(1, 256)
    my_value, my_logit, (my_hx, my_cx) = my_model.forward((state.unsqueeze(0), (hx, cx)))
    value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
    
    

    
    print("--- check output shape ---")
    my_value, my_logit, (my_hx, my_cx) = my_model.forward((state.unsqueeze(0), (hx, cx)))
    value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

    print(value.shape, my_value.shape)
    print(logit.shape, my_logit.shape)
    print(hx.shape, my_hx.shape)
    print(cx.shape, my_cx.shape)
    
    print("--- check output accuracy ---")

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

    my_value, my_logit, (my_hx, my_cx) = my_model.forward((state.unsqueeze(0), (hx, cx)))
    value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

    print(sum(sum(abs(value - my_value))))
    print(sum(sum(abs(logit - my_logit))))
    print(sum(sum(abs(hx - my_hx))))
    print(sum(sum(abs(cx - my_cx))))
    
    print("---- checkout model backward ----")


    loss = value + logit.sum()
    loss.backward()

    top_grad_logit = torch.ones(logit.shape) 
    top_grad_value = torch.ones(value.shape)
    bottom_grad = my_model.backward(top_grad_value, top_grad_logit)

    def eval(value1, value2):
        if value1.shape != value2.shape:
            print("error")
            return 
        
        print(torch.max(torch.abs((value1 - value2) / value2)))

    
    eval(my_model.actor_linear.grad_weight, model.actor_linear.weight.grad)
    eval(my_model.actor_linear.grad_bias, model.actor_linear.bias.grad)
    eval(my_model.critic_linear.grad_weight, model.critic_linear.weight.grad)
    eval(my_model.critic_linear.grad_bias, model.critic_linear.bias.grad)

    eval(my_model.lstm.grad_bias_hh, model.lstm.bias_hh.grad)
    eval(my_model.lstm.grad_bias_ih, model.lstm.bias_ih.grad)
    eval(my_model.lstm.grad_weight_ih, model.lstm.weight_ih.grad)
    eval(my_model.lstm.grad_weight_hh, model.lstm.weight_hh.grad)
'''
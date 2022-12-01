import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from algorithms import Algorithm

NN_AGENT_LOSSES = [[] for _ in range(12)]

class MLP(nn.Module):
    def __init__(self, input_size):
        self.size = input_size
        super().__init__()
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, x.shape[0]*x.shape[1])
        # print(f"x shape: {x.shape}")
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = torch.tanh(x)
        return x

class DeepLearning(Algorithm):
    def __repr__(self):
        return self.name
    def __init__(self):
        self.name = "NN"

    def setup(self, lr):
        # self.agent_networks = dict()
        # self.agent_optims = dict()
        self.model = None
        self.optim = None
        self.learning_rate = float(lr)

        def estimate_func(observations, true_state, incoming_comms):
            arr_shape = incoming_comms[0].shape
            v = np.empty(arr_shape) # assume they get at least one
            for agent in observations:
                v[agent] = true_state[agent] # ground truth
            for agent in range(arr_shape[0]):
                if agent not in observations:
                    # if agent not in self.agent_networks:
                        # self.agent_networks[agent] = MLP(len([c[agent] for c in incoming_comms]) * incoming_comms[0].shape[1])
                        # self.agent_optims[agent] = optim.SGD(params=self.agent_networks[agent].parameters(), lr=self.learning_rate, momentum=0.9)
                    if self.model is None:
                        self.model = MLP(len([c[agent] for c in incoming_comms]) * incoming_comms[0].shape[1])
                        self.optim = optim.SGD(params=self.model.parameters(), lr=self.learning_rate, momentum=0.9)

                    inp = torch.FloatTensor(np.array([c[agent] for c in incoming_comms]))
                    e = self.model(inp)

                    v[agent] = e.cpu().detach().numpy().flatten()
            return v

        def update_func(agent_id, state_estimate, loss_fn, alpha=self.learning_rate):
            x_tensor = torch.autograd.Variable(torch.Tensor(state_estimate), requires_grad=True)
            y = loss_fn(x_tensor)
            y.backward()
            # if agent_id in self.agent_optims:
            #     self.agent_optims[agent_id].step()
            #     self.agent_optims[agent_id].zero_grad()
            self.optim.step()
            self.optim.zero_grad()
            NN_AGENT_LOSSES[agent_id].append(y.item())
            grad = x_tensor.grad
            new_state = state_estimate
            new_state[agent_id] -= alpha * grad[agent_id].cpu().detach().numpy() # gradient descent step
            # self.learning_rate = 0.99*self.learning_rate    # learning rate decay
            return new_state

        def communicate_func(state_estimate):
            return state_estimate + np.random.normal(size=state_estimate.shape) # add Gaussian noise

        self.g = estimate_func
        self.f = update_func
        self.h = communicate_func

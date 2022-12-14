# Algorithm implementations

import numpy as np
from scipy.optimize import minimize, differential_evolution
import torch
from torch import nn

INDIVIDUAL_AGENT_LOSS = [[] for _ in range(12)]

BASE_ITER = 100

def softmax(x, T=1):
    """Compute softmax values for each sets of scores in x."""
    # Parameter T is the temperature
    e_x = np.exp((x - np.max(x))/T)
    return e_x / e_x.sum()

class Algorithm:
    # Generic wrapper for all algorithm classes

    def __init__(self):
        self.g = None # estimation function
        self.f = None # update function
        self.h = None # adversarial communication function

    def setup(self):
        # Generic gradient descent update and Gaussian noise communcation function
        # Overwrite this in the derived class if you want to change it

        def update_func(agent_id, state_estimate, loss_fn, alpha=self.alpha):
            x_tensor = torch.autograd.Variable(torch.Tensor(state_estimate), requires_grad=True)
            y = loss_fn(x_tensor)
            y.backward()
            INDIVIDUAL_AGENT_LOSS[agent_id].append(y.item())
            grad = x_tensor.grad
            new_state = state_estimate
            new_state[agent_id] -= alpha * grad[agent_id].cpu().detach().numpy() # gradient descent step

            return new_state
        self.f = update_func

        def communicate_func(state_estimate):
            return state_estimate + np.random.normal(size=state_estimate.shape) # add Gaussian noise
        self.h = communicate_func


class Baseline(Algorithm):

    def __repr__(self):
        return self.name

    def __init__(self):
        self.name = "Baseline"

    def setup(self, alpha, D_local):
        """
        Sets up the algorithm for a new test
        """
        self.alpha = float(alpha)
        self.D_local = D_local
        super().setup()

        def estimate_func(observations, true_state, incoming_comms):
            arr_shape = incoming_comms[0].shape
            v = np.empty(arr_shape) # assume they get at least one
            for agent in observations:
                v[agent] = true_state[agent] # ground truth
            # Use an average with the D highest and D smallest values removed
            for agent in range(arr_shape[0]):
                if agent not in observations:
                    if self.D_local:
                        for state_component in range(arr_shape[1]):
                            sorted_by_component = np.sort([c[agent][state_component] for c in incoming_comms])
                            v[agent][state_component] = np.mean(sorted_by_component[self.D_local:-self.D_local])
                    else:
                        v[agent] = np.mean([c[agent] for c in incoming_comms], axis=0)
            return v

        self.g = estimate_func

class ExpGaussianConverge(Algorithm):

    def __repr__(self):
        return self.name

    def __init__(self):
        self.name = "ExpGaussianConverge"

    def setup(self, alpha, gamma, T):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.T = float(T)
        self.p = None # vector of n values for weighting truthfulness
        super().setup()

        def estimate_func(observations, true_state, incoming_comms):
            n_i = len(incoming_comms) # number of communication partners
            arr_shape = incoming_comms[0].shape
            if self.p is None:
                self.p = np.zeros(n_i)

            v = np.empty(arr_shape) # assume they get at least one
            for agent in range(arr_shape[0]):
                if agent in observations:
                    v[agent] = true_state[agent] # ground truth
                else:
                    # Update historical truthfulness values using Gaussian approach
                    comms = np.array([c[agent] for c in incoming_comms])
                    mu = np.mean(comms, axis=0)
                    cov = np.cov(comms.T)
                    inv_cov = np.linalg.inv(cov+np.eye(arr_shape[1])*1e-9) # to avoid singularity
                    for m, c in enumerate(incoming_comms): # m is indexed relative to the mth communication partner
                        self.p[m] = (1-self.gamma)*self.p[m] - self.gamma*((c[agent]-mu) @ inv_cov @ (c[agent]-mu).T)

            weights = softmax(self.p, T) # temperatured softmax
            for agent in range(arr_shape[0]):
                if agent not in observations:
                    v[agent] = np.sum([weights[i]*incoming_comms[i][agent] for i in range(n_i)], axis=0) # weighted sum by truthfulness
            return v
        self.g = estimate_func

class CumulativeL2(Algorithm):

    def __repr__(self):
        return self.name

    def __init__(self):
        self.name = "CumulativeL2"

    def setup(self, alpha, T):
        self.alpha = float(alpha)
        self.T = float(T)
        self.p = None # vector of n values for weighting truthfulness
        super().setup()

        def estimate_func(observations, true_state, incoming_comms):
            n_i = len(incoming_comms) # number of communication partners
            arr_shape = incoming_comms[0].shape
            if self.p is None:
                self.p = np.zeros(n_i)

            v = np.empty(arr_shape) # assume they get at least one
            for agent in range(arr_shape[0]):
                if agent in observations:
                    v[agent] = true_state[agent] # ground truth
                else:
                    # Update historical truthfulness values using cumulative L2 norm approach
                    comms = np.array([c[agent] for c in incoming_comms])
                    mu = np.mean(comms, axis=0)
                    for m, c in enumerate(incoming_comms): # m is indexed relative to the mth communication partner
                        self.p[m] -= np.linalg.norm(c[agent] - mu)**2

            weights = softmax(self.p, T) # temperatured softmax
            for agent in range(arr_shape[0]):
                if agent not in observations:
                    v[agent] = np.sum([weights[i]*incoming_comms[i][agent] for i in range(n_i)], axis=0) # weighted sum by truthfulness
            return v
        self.g = estimate_func

class RLAgent(Algorithm):
    def __repr__(self):
        return self.name

    def __init__(self):
        self.name = "RLAgent"
        self.dqn = None

    def setup(self, alpha, beta, gamma):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.p = None
        T = 100
        super().setup()

        # SAME AS CL2 for now!
        def estimate_func(observations, true_state, incoming_comms):
            n_i = len(incoming_comms) # number of communication partners
            arr_shape = incoming_comms[0].shape
            if self.p is None:
                self.p = np.zeros(n_i)

            v = np.empty(arr_shape) # assume they get at least one
            for agent in range(arr_shape[0]):
                if agent in observations:
                    v[agent] = true_state[agent] # ground truth
                else:
                    # Update historical truthfulness values using cumulative L2 norm approach
                    comms = np.array([c[agent] for c in incoming_comms])
                    mu = np.mean(comms, axis=0)
                    for m, c in enumerate(incoming_comms): # m is indexed relative to the mth communication partner
                        self.p[m] -= np.linalg.norm(c[agent] - mu)**2

            weights = softmax(self.p, T) # temperatured softmax
            for agent in range(arr_shape[0]):
                if agent not in observations:
                    v[agent] = np.sum([weights[i]*incoming_comms[i][agent] for i in range(n_i)], axis=0) # weighted sum by truthfulness
            return v
        self.g = estimate_func

        def update_func(agent_id, state_estimate, loss_fn, alpha=self.alpha):
            # Feed full state information to Q network
            if not self.dqn:
                self.dqn = DQN(len(state_estimate), len(state_estimate[0]))
            
            def bellman_func(a):
                new_state = state_estimate
                new_state[agent_id] += a
                new_state = torch.Tensor(new_state)
                return self.alpha*np.linalg.norm(a)**2 + self.beta + self.gamma*loss_fn(new_state)

            minimizer = differential_evolution(bellman_func, bounds=[(-2,2),(-2,2)], maxiter=500)
            print(minimizer) # not minimizing for some reason
            opt_policy = minimizer.x
            opt_value = bellman_func(opt_policy)

            print(opt_policy)
            x_tensor = torch.autograd.Variable(torch.Tensor(state_estimate), requires_grad=True)
            bellman_loss = lambda x_tensor: (self.dqn(x_tensor) - opt_value)**2
            print(bellman_loss(x_tensor))
            bellman_loss(x_tensor).backward() # Q-network training

            print(opt_policy)
            return opt_policy

        self.f = update_func

        
class DQN(nn.Module):
    def __init__(self, n, m):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n*m, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, m),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
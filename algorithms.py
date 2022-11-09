# Algorithm implementations

import numpy as np
import torch

BASE_ITER = 100

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Algorithm:
    # Generic wrapper for all algorithm classes

    def __init__(self):
        self.g = None # estimation function
        self.f = None # update function
        self.h = None # adversarial communication function

class SimpleMean(Algorithm):
    name = "SimpleMean"

    def __repr__(self):
        return self.name
    
    def __init__(self, alpha):
        self.alpha = float(alpha)

        def estimate_func(observations, true_state, incoming_comms): 
            arr_shape = incoming_comms[0].shape
            v = np.empty(arr_shape) # assume they get at least one 
            for agent in observations:
                v[agent] = true_state[agent] # ground truth
            # Use simple mean of other communications (essentially setting D=0 in his algo)
            for agent in range(arr_shape[0]):
                if agent not in observations:
                    v[agent] = np.mean([c[agent] for c in incoming_comms])
            return v
        self.g = estimate_func

        def update_func(agent_id, state_estimate, loss_fn, alpha=self.alpha):
            x_tensor = torch.autograd.Variable(torch.Tensor(state_estimate), requires_grad=True)
            y = loss_fn(x_tensor)
            y.backward()
            grad = x_tensor.grad
            new_state = state_estimate
            new_state[agent_id] -= alpha * grad[agent_id].cpu().detach().numpy() # gradient descent step

            return new_state 
        self.f = update_func

        def communicate_func(state_estimate):
            return state_estimate + np.random.normal(size=state_estimate.shape) # add Gaussian noise
        self.h = communicate_func

class ExpGaussianConverge(Algorithm):
    name = "ExpGaussianConverge"

    def __repr__(self):
        return self.name
    
    def __init__(self, alpha, gamma):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.p = None # vector of n values for weighting truthfulness

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
                    mu = np.mean([c[agent] for c in incoming_comms], axis=0)
                    cov = np.cov(np.array([c[agent] for c in incoming_comms]).T)
                    inv_cov = np.linalg.inv(cov+np.eye(arr_shape[1])*1e-9) # to avoid singularity
                    for m, c in enumerate(incoming_comms): # m is indexed relative to the mth communication partner
                        self.p[m] = (1-self.gamma)*self.p[m] - 0.5*self.gamma*((c[m]-mu) @ inv_cov @ (c[m]-mu).T)

            for agent in range(arr_shape[0]):
                weights = softmax(self.p)
                print(weights)
                if agent not in observations:
                    v[agent] = np.sum([weights[i]*incoming_comms[i][agent] for i in range(n_i)], axis=0) # weighted sum by truthfulness
            return v
        self.g = estimate_func

        # Update and communication functions remain unchanged!

        def update_func(agent_id, state_estimate, loss_fn, alpha=self.alpha):
            x_tensor = torch.autograd.Variable(torch.Tensor(state_estimate), requires_grad=True)
            y = loss_fn(x_tensor)
            y.backward()
            grad = x_tensor.grad
            new_state = state_estimate
            new_state[agent_id] -= alpha * grad[agent_id].cpu().detach().numpy() # gradient descent step

            return new_state 
        self.f = update_func

        def communicate_func(state_estimate):
            return state_estimate + np.random.normal(size=state_estimate.shape) # add Gaussian noise
        self.h = communicate_func
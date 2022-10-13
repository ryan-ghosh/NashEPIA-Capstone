# Algorithm implementations

import numpy as np
import torch

BASE_ITER = 100

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
        self.alpha = alpha

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


"""
class BaseLine:


    def baseline(self, Network):
        N = len(Network.agents)
        # print('based')
        for k in range(BASE_ITER):

            print("running baseline algo")
            # print(Network.agents[0].id)

            for i in range(N):
                print(f"{Network.agents[i].id} communicates to    : ", [x.id for x in Network.c_graph.out_neighbors[i]])
            
            for i in range(N):
                print(f"{Network.agents[i].id} receives comms from: ", [x.id for x in Network.c_graph.in_neighbors[i]])

            for i in range(N):
                print(f"{Network.agents[i].id} observes: ", [x.id for x in Network.o_graph.out_neighbors[i]])

            # for i in range(N):
            # step 1 of Dian's algo
            Network.update_network_state()

            # for i in range(Network.agents.size):
            #     print(f'agent {i}: ', Network.agents[i].name)
            #     for j in range(Network.c_graph[i].neighbors.size):      # iterate through neighbours of ith agent
            #         # send communicated message from agent i to agent j
            #         # (i.e., send y_ji[k] where j is an outgoing neighbor to agent i)
            #         # and receive all messages from agent j to agent i
            #         # (i.e., receive y_ij[k] where j is an incoming neighbor to agent i)
            #         pass

            for m in range(Network.agents.size):
                n_m = Network.agents[0].state.size
                for q in range(n_m):
                    # need set T of truthful agents.
                    # for i in range(Network.T.size):
                    #
                    #   remove D highest and D smallest vals y_mq^ij that are larger and smaller than x_mq^i
                    #  y_mq^ij = agent j's communicated message to agent i about the qth comp of agent m's action
                    pass

if __name__ == "__main__":
    # quick sanity check
    N = 3
    M = 10
    
    its = State(np.random.rand(N,M))
    lfn = 1
    a1 = Agent("a1", int, np.random.rand(N,M), np.random.rand(N,M), lfn, 0,0)
    a2 = Agent("a2", int, np.random.rand(N,M), np.random.rand(N,M), lfn, 0,0)
    a3 = Agent("a3", int, np.random.rand(N,M), np.random.rand(N,M), lfn, 0,0)
    a4 = Agent("a4", int, np.random.rand(N,M), np.random.rand(N,M), lfn, 0,0)

    agents = np.array([a1, a2, a3, a4], dtype=Agent)

    c_adj =    [[0,1,0,0],
                [0,0,0,1],
                [0,1,0,1],
                [1,0,0,0]]

    o_adj =    [[0,1,1,0],
                [0,0,1,0],
                [1,0,0,0],
                [0,0,0,0]]

    g_c = DirectedGraph(agents, c_adj)
    g_o = DirectedGraph(agents, o_adj)

    net = Network(agents, its, g_c, g_o)

    bl = BaseLine()
    bl.baseline(net)
"""
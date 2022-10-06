from typing import Tuple
import numpy as np

TRUTHFUL = 1
ADVERSARIAL = 0

class State:
    ## not sure how we want this implemented right now, just leaving as position.
    def __init__(self, x:np.array):
        self.x = x
    
    def size(self):
      return self.x.shape[0]

class Agent:
    def __init__(self, name, type: int, init_state: np.array, loss_fn, f, h):
        self.type = type
        self.loss_fn = loss_fn
        self.state = State(np.array)
        self.name = name
        self.f = f
        if type == ADVERSARIAL:
          self.h = h
        else:
          self.h = lambda x : x # idk if this works lol, I've never used lambda func in python before

    def __repr__(self):
        return self.name

    def update_state(self):
        self.state.x = self.f(self.state.x, self.state.y)
        self.state.y = self.h(self.state.x, self.state.y)

    def send_message(self) -> State:
        # if self.type == ADVERSARIAL:
        #     ## send noise message around last state, can change to completely random if wanted
        #     n_x = np.random.normal(self.state.x, 1)
        #     n_y = np.random.normal(self.state.y, 1)
        #     return State(n_x, n_y)

        # return self.state
        return self.h(self.state)

class Network:
    def __init__(self, agents, init_true_state: State, c_graph=None, o_graph=None):
        if c_graph is None and o_graph is None:
            ## can use for random initialization of large networks
            pass

        self.c_graph = c_graph
        self.o_graph = o_graph
        self.agents = agents
        self.weights = np.ones((len(self.agents), 1))   ## weights for each agent
        self.true_state = init_true_state
        

    def update_network_state(self):
        ## something like this might be possible

        ## iterate over all possible agents
          ## call agent.update_states and agent.send_message and store values
          ## update network true state using ^
          ## update agent internal states using ^^
          ## send messages stored in ^^^
        pass
    
    """
    Some brief comments
    - I am not sure what the loss of an entire network means
    - Are we just using the sum of the loss of all agents? of all truthful agents? 
    - Either way, agent.loss_fn is a function of the 'true state', not the graphs
    """
    def compute_loss(self) -> float:
        loss = 0.0
        for agent in self.agents:
            loss += agent.loss_fn(self.true_state) ## passing all 3 as params cause not sure what we are using

        return loss

class NashEPIA:
    def __init__(self, network, algo):
        self.network = network
        self.solver = algo

    def run(epsilon) -> tuple:
        pass

import numpy as np

TRUTHFUL = 1
ADVERSARIAL = 0

class State:
    ## not sure how we want this implemented right now, just leaving as position.
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y

class Agent:
    def __init__(self, name, type: str, start_x: float, start_y: float, loss_fn, f, h):
        self.type = type
        self.loss_fn = loss_fn
        self.state = State(start_x, start_y)
        self.name = name
        self.f = f
        self.h = h

    def __repr__(self):
        return self.name

    def update_state(self):
        self.state.x = self.f(self.state.x, self.state.y)
        self.state.y = self.h(self.state.x, self.state.y)

    def send_message(self) -> State:
        if self.type == ADVERSARIAL:
            ## send noise message around last state, can change to completely random if wanted
            n_x = np.random.normal(self.state.x, 1)
            n_y = np.random.normal(self.state.y, 1)
            return State(n_x, n_y)

        return self.state

class Network:
    def __init__(self, agents, c_graph=None, o_graph=None):
        if c_graph is None and o_graph is None:
            ## can use for random initialization of large networks
            pass

        self.c_graph = c_graph
        self.o_graph = o_graph
        self.agents = agents
        self.weights = np.ones((len(self.agents), 1))   ## weights for each agent

    def update_network_state(self):
        pass

    def compute_loss(self) -> float:
        loss = 0.0
        for agent in self.agents:
            loss += agent.loss_fn(self.c_graph[agent], self.o_graph[agent], self.agents) ## passing all 3 as params cause not sure what we are using

        return loss

class NashEPIA:
    def __init__(self, network, algo):
        self.network = network
        self.solver = algo

    def run(epsilon) -> tuple[int, float]:
        pass

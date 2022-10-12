from typing import Tuple
import numpy as np

TRUTHFUL = 1
ADVERSARIAL = 0

class State:
    def __init__(self, x: np.array):
        self.x = x  ## n by m array (n is number of agents)
        self.n = self.x.shape[0]
        self.m = self.x.shape[1]

    def size(self):
      return self.x.shape[0]

class Agent:
    def __init__(self, id, type: int, init_estate: np.array, init_cstate: np.array, loss_fn, f, h):
        self.type = type
        self.id = id
        self.loss_fn = loss_fn
        self.e_state = State(init_estate)
        self.c_state = State(init_cstate)
        self.f = f
        if type == ADVERSARIAL:
          self.h = h
        else:
          self.h = lambda x : x

    def __repr__(self):
        return str(id)

    def get_agent_state(self):
        return self.state[self.id]

    def update_agent_state(self):
        self.state[self.id] = self.f(self.state.x)     ## how agent updates its estimate

    ## not sure how agent's state estimate should be updated

    def send_message(self) -> State:
        return self.h(self.state)

class DirectedGraph:
    def __init__(self, agents, adj_list):
        ## not sure how we want to design the graph, but I usually use an
        ## adjacency list, i.e. adj_list[i] is a list of vertix indices j s.t.
        ## the edge i->j exists in the graph

        self.V = len(agents)
        self.adj_list = adj_list

        ## determine neighbors of each vertex, i.e. neighbors[i] is a list
        ## of vertix indices j s.t. j->i is an edge in the graph
        self.out_neighbors = [[] for i in range(self.V)]
        self.in_neighbors = [[] for i in range(self.V)]

        for i in range(self.V):
            for j in range (self.V):
                if (self.adj_list[i][j] != 0):
                    self.out_neighbors[i].append(agents[j])
                    # print("appending", agents[j].id, "to self_neighbors ",i)
                if (self.adj_list[j][i] != 0):
                    self.in_neighbors[i].append(agents[j])

        
                

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

        for agent in self.agents:
            ## assuming agent name member is index here, can change if necassary later
            agent.update_state()
            message = agent.send_message()
            true_agent_state = agent.state
            agent_id = agent.id
            for neighbour in self.c_graph[agent_id]:
                neighbour.state[agent_id] = message

            self.true_state[agent_id] = true_agent_state

    def compute_loss(self) -> float:
        loss = 0.0
        for agent in self.agents:
            loss += agent.loss_fn(self.true_state)

        return loss

class NashEPIA:
    def __init__(self, network, algo):
        self.network = network
        self.solver = algo

    # def run(epsilon) -> tuple[float, int]:  ## returns loss and number of iterations
    #     pass

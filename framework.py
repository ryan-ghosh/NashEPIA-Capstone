from algorithms import *
import numpy as np
from matplotlib import pyplot as plt
import torch

TRUTHFUL = 1
ADVERSARIAL = 0

class Agent:
    def __init__(self, id: int, type: int, init_estate: np.array, loss_fn):
        # Attributes
        self.type = type
        self.id = id
        self.loss_fn = loss_fn
        self.e_state = init_estate

    def setup(self, g, f, h, G_o, G_c):
        # Functions (property of algorithm)
        self.g = g
        self.f = f
        self.h = h if self.type == ADVERSARIAL else lambda x: x
        self.c_state = h(self.e_state) # initialize first message

        # Neighbors (property of network)
        self.obs_neighbours = [i for i, v in enumerate(G_o[self.id]) if v]
        self.comm_neighbours = [i for i, v in enumerate(G_c[self.id]) if v]

    def __repr__(self):
        return self.id # integer from 0...n-1

    def update_state(self, true_state, all_communications):
        comm_messages = [all_communications[agent] for agent in self.comm_neighbours]
        self.e_state = self.f(
            self.id,
            self.g(self.obs_neighbours, true_state, comm_messages), 
            self.loss_fn
        )
        self.c_state = self.h(self.e_state)

class Network:
    def __init__(self, agents, init_true_state, G_c, G_o):
        self.G_c = G_c
        self.G_o = G_o
        self.agents = agents
        self.true_state = init_true_state

    def iterate(self):
        all_communications = [ agent.c_state for agent in self.agents ]
        for agent in self.agents:
            agent.update_state(self.true_state, all_communications)
        for agent in self.agents:
            self.true_state[agent.id] = agent.e_state[agent.id]

class NashEPIA:
    def __init__(self, network: Network, algo: Algorithm):
        self.network = network
        self.solver = algo

    def setup(self):
        for agent in self.network.agents:
            agent.setup(self.solver.g, self.solver.f, self.solver.h, self.network.G_c, self.network.G_o)

    def run(self, epsilon, max_iter = 10000):  
        '''
        Returns number of iterations to convergence with maximum L2-difference (Frobenius norm) between states epsilon
        Also returns the final states for plotting the Nash equilibrium
        '''

        distance_vector, all_states = [], []

        last_state = np.copy(self.network.true_state)
        iterations = 0
        while iterations < max_iter:
            iterations += 1
            self.network.iterate()
            frob_distance = np.linalg.norm( last_state.flatten() - self.network.true_state.flatten() )
            distance_vector.append(frob_distance)
            all_states.append(last_state)
            print(f"Iteration {iterations}: L2-movement since last iter: {frob_distance}")
            if frob_distance < epsilon: # convergence with
                return (iterations, distance_vector, all_states, self.network.true_state)
            last_state = np.copy(self.network.true_state)
        
        print(f"Did not converge within max iterations of {max_iter}")
        return (iterations, distance_vector, all_states, self.network.true_state)


if __name__ == "__main__":
    # Basic test - 3 robots who just want to converge to each other
    # All are truthful and travel in only one dimension (Nash eq'm is all at the same spot)
    n = 3
    m = 2 # two dimensional
    loss_fn = lambda state: sum([ torch.norm(x1-x2)**2 for x1 in state for x2 in state ] ) 
    init_states = np.random.normal(0, 16, size=(n,m)) # initial positions N(0,16)
    agents = [ Agent(i, TRUTHFUL, np.copy(init_states), loss_fn) for i in range(3) ]
    print(f"Starting positions: {init_states}\n\n")

    G_c = np.array([ [0,1,1], [1,0,1], [1,1,0] ]) # fully connected
    G_o = np.eye(3) # only self-observational
    net = Network(agents, init_states, G_c, G_o)
    algo = SimpleMean(alpha=0.01)

    # Run test
    nepia = NashEPIA(net, algo)
    nepia.setup()
    results = nepia.run(epsilon=1e-6)
    print(f"\n\n Results: {results} \n\n")

    # Loss plot
    '''
    plt.plot([i for i in range(results[0])], results[1])
    plt.title(f"{n} Robots in {m}D Space Converging to a Nash Eq'm")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.show()
    '''

    # Realtime plot
    plt.clf()
    plt.axis([-10, 10, -10, 10])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Realtime Dynamics of the {n} Robot System")
    for i in range(results[0]):
        plt.scatter(results[2][i][0][0], results[2][i][0][1], color="r")
        plt.scatter(results[2][i][1][0], results[2][i][1][1], color="b")
        plt.scatter(results[2][i][2][0], results[2][i][2][1], color="g")
        plt.pause(0.05)

    plt.show()
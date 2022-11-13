from algorithms import *
from copy import deepcopy
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

    def setup(self, algo, params, G_o, G_c):
        # Functions (property of algorithm)
        algo.setup(*params)
        self.g = algo.g
        self.f = algo.f
        self.h = algo.h if self.type == ADVERSARIAL else lambda x: x
        self.c_state = algo.h(self.e_state) # initialize first message

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
    def __init__(self, agents, init_true_state, G_c, G_o, adversaries):
        self.G_c = G_c
        self.G_o = G_o
        self.agents = agents
        self.true_state = init_true_state
        self.adversaries = adversaries
        self.truthful = [i for i in range(len(agents)) if i not in adversaries]

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

    def setup(self, params):
        for agent in self.network.agents:
            agent.setup(deepcopy(self.solver), params, self.network.G_o, self.network.G_c)

    def run(self, epsilon, max_iter = 20000):
        '''
        Returns number of iterations to convergence with maximum L2-difference (Frobenius norm) between states epsilon
        Also returns the final states for plotting the Nash equilibrium
        '''

        all_states, all_truthful_states = [], []

        last_state = np.copy(self.network.true_state)
        iterations = 0
        while iterations < max_iter:
            iterations += 1
            self.network.iterate()
            # Only take equilibrium of truthful agents
            last_state_truthful = last_state[self.network.truthful]
            current_state_truthful = self.network.true_state[self.network.truthful]
            frob_distance = np.linalg.norm( last_state_truthful.flatten() - current_state_truthful.flatten() )
            all_states.append(last_state)
            all_truthful_states.append(last_state_truthful)

            if iterations % 100 == 0:
                # print(f"Iteration: {iterations}, Last L2 Change: {frob_distance}")
                pass

            if frob_distance < epsilon: # convergence with
                #print(f"Terminated on iteration: {iterations}, Last L2 Change: {frob_distance}")
                # Compute distance vector from states for truthful agents
                distance_vector = [ np.linalg.norm(s-current_state_truthful) for s in all_truthful_states ]
                return (iterations, distance_vector, all_states, self.network.true_state)

            last_state = np.copy(self.network.true_state)

        print(f"Did not converge within max iterations of {max_iter}")
        distance_vector = [ np.linalg.norm(s-current_state_truthful) for s in all_truthful_states ]
        return (iterations, distance_vector, all_states, self.network.true_state)

# Algorithm implementations

import numpy as np
from framework import State, Agent, Network

BASE_ITER = 100

class BaseLine:
    name = "BaseLine"

    def __repr__(self):
        return self.name

    def baseline(self, Network):
        # print('based')
        for k in range(BASE_ITER):

            for i in range(Network.agents.size):
                print(f'agent {i}: ', Network.agents[i].name)
                for j in range(Network.c_graph[i].neighbors.size):      # iterate through neighbours of ith agent
                    # send communicated message from agent i to agent j
                    # (i.e., send y_ji[k] where j is an outgoing neighbor to agent i)
                    # and receive all messages from agent j to agent i
                    # (i.e., receive y_ij[k] where j is an incoming neighbor to agent i)
                    pass

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
    its = State(np.random.rand(10))
    lfn = 1
    a1 = Agent('a1', int, State(np.random.rand(10)), lfn, 0,0)
    a2 = Agent('a2', int, State(np.random.rand(10)), lfn, 0,0)
    a3 = Agent('a3', int, State(np.random.rand(10)), lfn, 0,0)

    agents = np.array([a1, a2, a3], dtype=Agent)

    net = Network(agents, its)

    bl = BaseLine()
    bl.baseline(net)

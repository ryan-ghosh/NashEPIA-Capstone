# Test convergence in example graphs

from algorithms import *
from NN import DeepLearning
from framework import *

import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import time


ALGORITHMS = {
    "Baseline": Baseline,
    "CumulativeL2": CumulativeL2,
    "ExpGaussianConverge": ExpGaussianConverge,
    "NN": DeepLearning
}

class TestNashEPIA:
    tests = [
        # "tests/simple_test.json",
        # "tests/simple_test_adversary.json",
        # "tests/5n_2d_1a_1.json",
        # "tests/dian.json",
        "tests/dian10adv.json"
    ]

    def __init__(self, algo, n, vis, loss_plot, dist_plot, params):
        self.algo = algo
        self.n = n
        self.visualize = vis
        self.params = params
        self.loss_plot = loss_plot
        self.dist_plot = dist_plot
        if params:
            for i,p in enumerate(params):
                self.params[i] = float(p) # convert strings to float arg
        if self.algo.name == "Baseline":
            self.params.append(0)

    def run(self):
        total_tests = len(self.tests)
        novel_win_iter = 0.0
        failed_tests = list()

        start_time = time.time()
        all_iter = [[]]*total_tests
        for k in range(self.n):
            for i, testpath in enumerate(self.tests):
                # Extract test info before setting up the algorithm
                test, eps, max_iter, D_local, NE = self.parse_test(testpath)

                # Instantiate Network and run
                novel_Nash = NashEPIA(test, self.algo)
                if self.algo.name == "Baseline":
                    self.params[-1] = D_local # update locality metric

                novel_Nash.setup(self.params)
                novel_iter, novel_dist, novel_states, novel_final_state = novel_Nash.run(eps, max_iter)

                if self.visualize and len(test.agents[0].e_state[0]) == 2:
                    # Realtime plot of 4 robot system
                    xmin = np.min(np.array(novel_states)[:,:,0])
                    xmax = np.max(np.array(novel_states)[:,:,0])
                    rx = xmax-xmin
                    ymin = np.min(np.array(novel_states)[:,:,1])
                    ymax = np.max(np.array(novel_states)[:,:,1])
                    ry = ymax-ymin

                    plt.clf()
                    plt.axis([xmin - 0.1*rx, xmax + 0.1*rx,
                              ymin - 0.1*ry, ymax + 0.1*ry])
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.title(f"Realtime Dynamics of the Robot System")
                    for robot in range(len(test.agents)):
                        plt.plot([novel_states[i][robot][0] for i in range(novel_iter)], [novel_states[i][robot][1] for i in range(novel_iter)],
                            '--' if test.agents[robot].type == ADVERSARIAL else '-')
                        plt.scatter(novel_final_state[robot][0], novel_final_state[robot][1], marker="*", s=20)
                    plt.show()

                if self.loss_plot:
                    plt.title("Distance to the stationary point as a function of iterations")
                    plt.plot([i for i in range(novel_iter)], novel_dist)
                    plt.xlabel("Iteration")
                    plt.ylabel("$||x-\\bar{x}||$")
                    plt.yscale("log")
                    plt.show()

                    if NE is not None:
                        dist_vec = [ np.linalg.norm(s-NE) for s in novel_states ]
                        plt.title("Distance to the Nash equilibrium as a function of iterations")
                        plt.plot([i for i in range(novel_iter)], dist_vec)
                        plt.xlabel("Iteration")
                        plt.ylabel("$||x-x*||$")
                        plt.yscale("log")
                        plt.show()

                all_iter[i].append(novel_iter)

        test_time = time.time() - start_time

        for i in range(total_tests):
            failed_tests = 0
            for iter in all_iter[i]:
                if iter == max_iter:
                    failed_tests += 1
            print(f'Test {self.tests[i]}: {self.algo} average iterations: {sum(all_iter[i]) / self.n} across {self.n} tests')
            print(f'Test {self.tests[i]}: {self.algo} number of failed tests: {failed_tests} across {self.n} tests')

        if self.dist_plot:
            for i, testpath in enumerate(self.tests):
                plt.title(testpath)
                plt.xlabel("Iterations to Convergence")
                plt.ylabel("Frequency")
                plt.hist(all_iter[i])
                plt.show()

        print(f'Total wall time elapsed for all tests: {test_time} s')

    def parse_test(self, test_path):
        with open(test_path, 'r') as f:
            test = json.loads(f.read())

        # parse out info
        n, m = test["num_agents"], test["state_dim"]
        G_c, G_o = np.array(test["G_c"]), np.array(test["G_o"])

        # state initialization
        if test["random_init_state"]:
            dist = test["init_state_distribution"]
            params = test["init_state_params"]
            if dist == "normal":
                init_states = np.random.normal(params[0], params[1], size=(n,m))
            elif dist == "uniform":
                init_states = np.random.uniform(params[0], params[1], size =(n,m))
        else:
            init_states = np.array(test["deterministic_init_state"])

        # Current set up adds standard normal noise to adversary output
        agents = []
        adversaries = []
        for i in range(n):
            loss_fn = self.generate_loss_fn(test, i)
            if str(i) in test["adversaries"]:
                agents.append(Agent(i, ADVERSARIAL, np.copy(init_states), loss_fn))
                adversaries.append(i)
            else:
                agents.append(Agent(i, TRUTHFUL, np.copy(init_states), loss_fn))

        # create Network object
        NE = np.array(test["NE"]) if "NE" in test else None
        return Network(agents, init_states, G_c, G_o, adversaries), test["eps"], test["max_iter"], test["D_local"], NE

    def generate_loss_fn(self, test, id):
        # extract loss_fn
        if str(id) in test["adversaries"]:
            loss_fn = test["adversaries"][str(id)]["loss_fn"]
        else:
            loss_fn = test["loss_fn"]

        # generate loss_fn accordingly
        if loss_fn == "dian":
            # as defined in Dian's paper...
            def f(state):
                cost = 0.5*torch.norm(torch.mean(state, axis=0) - torch.Tensor(test["Q"]))**2 # 'a' term
                # r_i term: note all relative distances defined relative to the 0th agent
                d0j = test["d0j"]

                if id == 0: # relative position to all other agents
                    for j in range(1, test["num_agents"]):
                        dij = torch.Tensor(d0j[j])
                        if len(dij) != 0: cost += 0.5*torch.norm(state[id] - state[j] - torch.Tensor(dij))**2
                elif len(d0j[id]) != 0: # check if relative position was specified for this agent
                    cost += 0.5*torch.norm(state[id] - state[0] + torch.Tensor(d0j[id]))**2 # x_id - x_0 penalty

                    # NEED TO DEBUG THIS
                    # for j in range(1, test["num_agents"]): # x_id - x_j penalty for remaining agents
                    #     if id != j and len(d0j[j]) != 0:
                    #         dij = torch.Tensor(d0j[j]) - torch.Tensor(d0j[id]) # dij = d0j-d0i
                    #         cost += 0.5*torch.norm(state[id] - state[0] - dij)**2


                return cost
            return f
        else:
            return eval(loss_fn)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for NashEPIA testing')
    parser.add_argument("--algo", required=True, help="Algorithm to test")
    parser.add_argument("--algo_args", default=None, help="arguments for algorithm", nargs='*')
    parser.add_argument("--num_repeat", required=True, help="number of times to run each test")
    parser.add_argument("--visualize", default=False, help="will show a plot of convergence")
    parser.add_argument("--loss_plot", default=False, help="Plot loss verse iterations")
    parser.add_argument("--dist_plot", default=False, help="Plot the distribution of iterations for each test")

    args = parser.parse_args()
    novel_algo = args.algo
    params = args.algo_args
    num_repeat = int(args.num_repeat)
    vis = args.visualize
    loss_plot = bool(args.loss_plot)
    dist_plot = bool(args.dist_plot)

    tester = TestNashEPIA(ALGORITHMS[novel_algo](), num_repeat, vis, loss_plot, dist_plot, list(params))

    tester.run()
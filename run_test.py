# Test convergence in example graphs

from algorithms import *
from framework import *

import argparse
import numpy as np
import matplotlib.pyplot as plt
import json


ALGORITHMS = { # "baseline": BaseLine,
              "SimpleMean": SimpleMean}

class TestNashEPIA:
    tests = [
        "tests/simple_test.json"
    ]

    def __init__(self, algo):
        self.algo = algo

    def run(self):
        # bl_model = BaseLine()
        total_tests = len(self.tests)
        novel_win_iter = 0.0
        failed_tests = list()

        for i, testpath in enumerate(self.tests):
            test, eps, max_iter = self.parse_test(testpath)

            # baseline_Nash = NashEPIA(test, bl_model)
            # baseline_Nash.setup()
            novel_Nash = NashEPIA(test, self.algo)
            novel_Nash.setup()
            # bl_iter, bl_dist, bl_states, bl_true_states = baseline_Nash.run() <-- need to fill some values here
            novel_iter, novel_dist, novel_states, novel_true_states = novel_Nash.run(eps, max_iter)

            # if novel_iter < bl_iter:
            #     novel_win_iter += 1
            # else:
            #     failed_tests.append(i)

            ## Not sure how much we care about loss in comparisons since they will both converge
            # print(f'Test {i}: {self.algo} iterations: {novel_iter} {bl_model} iterations: {bl_iter}')
            print(f'Test {i}: {self.algo} iterations: {novel_iter}')
        print(f'{self.algo} had less iterations for {novel_win_iter/total_tests}% of tests compared to the baseline model')
        print(f'Failed tests: {failed_tests}')
    
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

        # TODO: figure out how to pass in noisey adveraries
        agents = []
        for i in range(n):
            if str(i) in test["adversaries"]:
                agents.append(Agent(i, ADVERSARIAL, np.copy(init_states), 
                    eval(test["adversaries"][str(i)]["loss_fn"])))
            else:
                agents.append(Agent(i, TRUTHFUL, np.copy(init_states), 
                    loss_fn))

        # create Network object
        return Network(agents, init_states, G_c, G_o), test["eps"], test["max_iter"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for NashEPIA testing')
    parser.add_argument("--algo", default=None, help="Algorithm to test")
    parser.add_argument("--algo_args", default=None, help="arguments for algorithm", nargs='*')

    args = parser.parse_args()
    novel_algo = args.algo
    params = args.algo_args

    if params:
        tester = TestNashEPIA(ALGORITHMS[novel_algo](eval(*params)))
    else:
        tester = TestNashEPIA(ALGORITHMS[novel_algo]())

    tester.run()
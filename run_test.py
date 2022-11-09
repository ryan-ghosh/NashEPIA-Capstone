# Test convergence in example graphs

from algorithms import *
from framework import *

import argparse
import numpy as np
import matplotlib.pyplot as plt
import json


ALGORITHMS = { 
    #"SimpleMean": SimpleMean,
    "ExpGaussianConverge": ExpGaussianConverge
}

class TestNashEPIA:
    tests = [
        #"tests/simple_test.json",
        "tests/simple_test_adversary.json"
    ]

    def __init__(self, algo, n, vis):
        self.algo = algo
        self.n = n
        self.visualize = vis

    def run(self):
        total_tests = len(self.tests)
        novel_win_iter = 0.0
        failed_tests = list()

        total_iter = [0]*total_tests
        for k in range(self.n):
            for i, testpath in enumerate(self.tests):
                test, eps, max_iter = self.parse_test(testpath)

                # baseline_Nash = NashEPIA(test, bl_model)
                # baseline_Nash.setup()
                novel_Nash = NashEPIA(test, self.algo)
                novel_Nash.setup()
                # bl_iter, bl_dist, bl_states, bl_true_states = baseline_Nash.run() <-- need to fill some values here
                novel_iter, novel_dist, novel_states, novel_final_state = novel_Nash.run(eps, max_iter)
                
                if self.visualize:
                    # Realtime plot of 4 robot system
                    plt.clf()
                    plt.axis([-20, 20, -20, 20])
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.title(f"Realtime Dynamics of the Robot System")
                    plt.plot([novel_states[i][0][0] for i in range(novel_iter)], [novel_states[i][0][1] for i in range(novel_iter)], color="r")
                    plt.plot([novel_states[i][1][0] for i in range(novel_iter)], [novel_states[i][1][1] for i in range(novel_iter)], color="g")
                    plt.plot([novel_states[i][2][0] for i in range(novel_iter)], [novel_states[i][2][1] for i in range(novel_iter)], color="b")
                    plt.plot([novel_states[i][3][0] for i in range(novel_iter)], [novel_states[i][3][1] for i in range(novel_iter)], color="k")

  
                    plt.scatter(novel_final_state[0][0], novel_final_state[0][0], color="r", marker="*", s=20)
                    plt.scatter(novel_final_state[1][0], novel_final_state[1][1], color="b", marker="*", s=20)
                    plt.scatter(novel_final_state[2][0], novel_final_state[2][1], color="g", marker="*", s=20)
                    plt.scatter(novel_final_state[3][0], novel_final_state[3][1], color="k", marker="*", s=20)

                plt.show()

                total_iter[i] += novel_iter
                print(novel_states)
                # if novel_iter < bl_iter:
                #     novel_win_iter += 1
                # else:
                #     failed_tests.append(i)

                ## Not sure how much we care about loss in comparisons since they will both converge
                # print(f'Test {i}: {self.algo} iterations: {novel_iter} {bl_model} iterations: {bl_iter}')
        
        for i in range(total_tests): 
            print(f'Test {i}: {self.algo} average iterations: {total_iter[i] / self.n} across {self.n} tests')

        #print(f'{self.algo} had less iterations for {novel_win_iter/total_tests}% of tests compared to the baseline model')
        #print(f'Failed tests: {failed_tests}')
    
    def parse_test(self, test_path):
        with open(test_path, 'r') as f:
            test = json.loads(f.read())

        # parse out info
        n, m = test["num_agents"], test["state_dim"]
        G_c, G_o = np.array(test["G_c"]), np.array(test["G_o"])
        loss_fn = eval(test["loss_fn"])

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
            if str(i) in test["adversaries"]:
                agents.append(Agent(i, ADVERSARIAL, np.copy(init_states), 
                    eval(test["adversaries"][str(i)]["loss_fn"])))
                adversaries.append(i)
            else:
                agents.append(Agent(i, TRUTHFUL, np.copy(init_states), 
                    loss_fn))

        # create Network object
        return Network(agents, init_states, G_c, G_o, adversaries), test["eps"], test["max_iter"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for NashEPIA testing')
    parser.add_argument("--algo", required=True, help="Algorithm to test")
    parser.add_argument("--algo_args", default=None, help="arguments for algorithm", nargs='*')
    parser.add_argument("--num_repeat", required=True, help="number of times to run each test")
    parser.add_argument("--visualize", default=False, help="will show a plot of convergence")

    args = parser.parse_args()
    novel_algo = args.algo
    params = args.algo_args
    num_repeat = int(args.num_repeat)
    vis = args.visualize

    if params:
        tester = TestNashEPIA(ALGORITHMS[novel_algo](*params), num_repeat, vis)
    else:
        tester = TestNashEPIA(ALGORITHMS[novel_algo](), num_repeat, vis)

    tester.run()
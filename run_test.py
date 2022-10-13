# Test convergence in example graphs

import argparse
from algorithms import *
from framework import NashEPIA

import numpy as np
import matplotlib.pyplot as plt


ALGORITHMS = {"baseline": BaseLine}

class TestNashEPIA:
    tests = [
        ## write test networks here
    ]

    def __init__(self, algo):
        self.algo = algo

    def run(self):
        bl_model = BaseLine()
        total_tests = len(self.tests)
        novel_win_iter = 0.0
        failed_tests = list()

        for i, test in enumerate(self.tests):
            baseline_Nash = NashEPIA(test, bl_model)
            novel_Nash = NashEPIA(test, self.algo)
            baseline_loss, baseline_iter = baseline_Nash.run()
            novel_loss, novel_iter = novel_Nash.run()

            if novel_iter < baseline_iter:
                novel_win_iter += 1
            else:
                failed_tests.append(i)

            ## Not sure how much we care about loss in comparisons since they will both converge
            print(f'Test {i}: {self.algo} loss: {novel_loss} {bl_model} loss: {baseline_loss}')

        print(f'{self.algo} had less iterations for {novel_win_iter/total_tests}% of tests compared to the baseline model')
        print(f'Failed tests: {failed_tests}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for NashEPIA testing')
    parser.add_argument("--algo", default=None, help="Algorithm to test")
    parser.add_argument("--algo_args", default=None, help="arguments for algorithm", nargs='*')

    args = parser.parse_args()
    novel_algo = args.algo
    params = args.algo_args

    assert novel_algo
    if params:
        tester = TestNashEPIA(ALGORITHMS[novel_algo](*params))
    else:
        tester = TestNashEPIA(ALGORITHMS[novel_algo]())

    tester.run()

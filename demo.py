import os
from time import sleep

if __name__ == "__main__":

    print("*********************************")
    print("**  Running Baseline Algorithm **")
    print("*********************************")
    os.system("python3 run_test.py --algo Baseline --algo_args 0.025 --num_repeat 1 --visualize 1 --loss_plot 1")    
    sleep(2)

    print("*************************************")
    print("**  Running CumulativeL2 Algorithm **")
    print("*************************************")
    os.system("python3 run_test.py --algo CumulativeL2 --algo_args 0.025 100 --num_repeat 1 --visualize 1 --loss_plot 1")
    sleep(2)

    print("**********************************")
    print("**  Running NeuralNet Algorithm **")
    print("**********************************")
    os.system("python3 run_test.py --algo NN --algo_args 0.025 --num_repeat 1 --visualize 1 --loss_plot 1")

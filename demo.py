import os
from time import sleep

if __name__ == "__main__":

    for i in range(2):
        print("\n\n")
        if i:
            print(f"Test Case {i+1}: 6 truthful drones, 6 adversarial drones with Gaussian noise")
            print("\n")
        else:
            print(f"Test Case {i+1}: 9 truthful drones, 3 adversarial drones with Gaussian noise")
            print("\n")
            
        print("*********************************")
        print("**  Running Baseline Algorithm **")
        print("*********************************")
        os.system(f"python3 run_test.py --algo Baseline --algo_args 0.025 --num_repeat 1 --visualize 1 --loss_plot 1 --test_index {i}")    
        sleep(2)

        print("*************************************")
        print("**  Running CumulativeL2 Algorithm **")
        print("*************************************")
        os.system(f"python3 run_test.py --algo CumulativeL2 --algo_args 0.025 100 --num_repeat 1 --visualize 1 --loss_plot 1 --test_index {i}")
        sleep(2)

        print("**********************************")
        print("**  Running NeuralNet Algorithm **")
        print("**********************************")
        os.system(f"python3 run_test.py --algo NN --algo_args 0.025 --num_repeat 1 --visualize 1 --loss_plot 1 --test_index {i}")



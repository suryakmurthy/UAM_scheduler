from task_gen_MDP import task_gen_mdp
from scheduler import scheduler
from UAM_MDP import uam_mdp
from scheduler_UAM import scheduler_uam
from task import task
import json
import os.path
from os.path import abspath
import sys

def check_Int (input):
    output = input
    if len(output) == 0:
        finished = True
    else:
        finished = False
    while not finished:
        try:
            int(output)
            finished = True

        except ValueError:
            output = raw_input("Not an Int. Please try again: ")
            if len(output) == 0:
                finished = True
            else:
                finished  = False
    return output
if __name__ == "__main__":
    print("Enter Values at Prompts")
    in_val = int(raw_input("Enter Number of Tasks in the System (Int): "))
    # in_val = check_Int(in_val)
    task_list = []
    hard_val = False
    for task_index in range(0, in_val):
        in_c = eval(raw_input("Enter Computation Probability Distribution for Task " + str(task_index) +": "))
        in_d = int(raw_input("Enter Deadline for Task " + str(task_index) +": "))
        in_a = eval(raw_input("Enter Inter-Arrival Probability Distribution for Task " + str(task_index) +": "))
        in_h = raw_input("Enter Type of Task  " + str(task_index) +" (h/s): ")
        if in_h == "h":
            hard_bool = True
            hard_val = True
        else:
            hard_bool = False
        new_task = task(in_c, in_d, in_a, hard=hard_bool)
        task_list.append(new_task)
    type_str = raw_input("Do you want a non-preemptible MDP? (y/n): ")
    if type_str == "y":
        MDP_obj = uam_mdp(task_list)
    else:
        MDP_obj = task_gen_mdp(task_list)
    depth = int(raw_input("Enter depth of MDP Generation (Int): "))
    MDP_obj.generate_MDP(depth)
    if type_str == "y":
        sched = scheduler_uam(MDP_obj)
    else:
        sched = scheduler(MDP_obj)
    num_samples = int(raw_input("Enter Number of Samples (Int): "))
    if hard_val:
        prob_c, prob_a = sched.hard_task_learning(0.1, 0.1, num_samples=num_samples)
    else:
        prob_c, prob_a = sched.soft_task_learning(0.1, 0.1, num_samples=num_samples)
    sched.make_estimate_MDP(depth)
    conv_val = raw_input("Enter Convergence Parameter for Value Iteration (Float): ")
    pol_1 = sched.optimal_policy(conv_val)
    print("Value Iteration Optimal Policy: ")
    for state in sched.estimate_MDP.state_list_scheduler:
        for job_1 in state:
            if job_1 == "Terminal":
                print job_1,
            else:
                print job_1.return_data(),
        print(pol_1[state])
    num_ep = int(raw_input("Enter Number of Episodes for Testing (Int): "))
    reward = sched.test_optimal_policy(pol_1, num_ep=num_ep)
    print("Value Iteration Reward Output: ", reward)
    MCTS_bool = raw_input("Would you like to run MCTS? (y/n): ")
    if MCTS_bool == "y":
        MCTS_ep = int(raw_input("Enter Number of episodes for MCTS Testing (Int): "))
        MCTS_sample = int(raw_input("Enter Number of samples for MCTS Testing (Int): "))
        MCTS_depth = int(raw_input("Enter depth for MCTS Testing (Int): "))
        EDF = raw_input("Would you like to run Earliest-Deadline-First? (y/n): ")
        if EDF == "y":
            MCTS_rew = sched.test_MCTS_policy(depth=MCTS_depth, num_samples=MCTS_sample, num_ep=MCTS_ep, rand=False)
        else:
            MCTS_rew = sched.test_MCTS_policy(depth=MCTS_depth, num_samples=MCTS_sample, num_ep=MCTS_ep, rand=True)
        print("MCTS Reward Output: ", MCTS_rew)
    print("Process Completed")
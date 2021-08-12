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
    print("Enter Values at Prompts (Press Enter for Default Values)")
    in_val = raw_input("Enter Number of Tasks in the System (Int): ")
    if in_val:
        in_val = int(in_val)
    else:
        in_val = 2
    task_list = []
    hard_val = False
    op_val = raw_input("Do you want to use the default task configuration? (y/n): ")
    if not op_val:
        op_val = "y"
    if op_val == "n":
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
    else:
        task_1 = task({1 : 1.0}, 2, {3: 1.0}, hard=True)
        task_2 = task({1 : 0.4, 2 : 0.5}, 3, {4: 1.0}, hard=False)
        hard_val = True
        task_list = [task_1, task_2]
    type_str = raw_input("Do you want a non-preemptible MDP? (y/n): ")
    if not type_str:
        type_str = "n"
    if type_str == "y":
        MDP_obj = uam_mdp(task_list)
    else:
        MDP_obj = task_gen_mdp(task_list)
    depth = raw_input("Enter depth of MDP Generation (Int): ")
    if depth:
        depth = int(depth)
    else:
        depth = 100
    MDP_obj.generate_MDP(depth)
    if type_str == "y":
        sched = scheduler_uam(MDP_obj)
    else:
        sched = scheduler(MDP_obj)
    num_samples = raw_input("Enter Number of Samples (Int): ")
    if num_samples:
        num_samples = int(num_samples)
    else:
        num_samples = 100
    if hard_val:
        prob_c, prob_a = sched.hard_task_learning(0.1, 0.1, num_samples=num_samples)
    else:
        prob_c, prob_a = sched.soft_task_learning(0.1, 0.1, num_samples=num_samples)
    sched.make_estimate_MDP(depth)
    conv_val = raw_input("Enter Convergence Parameter for Value Iteration (Float): ")
    if not conv_val:
        conv_val = 0.01
    pol_1 = sched.optimal_policy(conv_val)
    print("Value Iteration Optimal Policy: ")
    for state in sched.estimate_MDP.state_list_scheduler:
        for job_1 in state:
            if job_1 == "Terminal":
                print job_1,
            else:
                print job_1.return_data(),
        print(pol_1[state])
    num_ep = raw_input("Enter Number of Episodes for Testing (Int): ")
    if num_ep:
        num_ep = int(num_ep)
    else:
        num_ep = 10
    reward = sched.test_optimal_policy(pol_1, num_ep=num_ep)
    print("Value Iteration Reward Output: ", reward)
    MCTS_bool = raw_input("Would you like to run MCTS? (y/n): ")
    if not MCTS_bool:
        MCTS_bool = "y"
    if MCTS_bool == "y":
        MCTS_ep = raw_input("Enter Number of episodes for MCTS Testing (Int): ")
        if MCTS_ep:
            MCTS_ep = int(MCTS_ep)
        else:
            MCTS_ep = 10
        MCTS_sample = raw_input("Enter Number of samples for MCTS Testing (Int): ")
        if MCTS_sample:
            MCTS_sample = int(MCTS_ep)
        else:
            MCTS_sample = 10
        MCTS_depth = raw_input("Enter depth for MCTS Testing (Int): ")
        if MCTS_depth:
            MCTS_depth = int(MCTS_ep)
        else:
            MCTS_depth = 10
        EDF = raw_input("Would you like to run Earliest-Deadline-First? (y/n): ")
        if not EDF:
            EDF = "y"
        if EDF == "y":
            MCTS_rew = sched.test_MCTS_policy(depth=MCTS_depth, num_samples=MCTS_sample, num_ep=MCTS_ep, rand=False)
        else:
            MCTS_rew = sched.test_MCTS_policy(depth=MCTS_depth, num_samples=MCTS_sample, num_ep=MCTS_ep, rand=True)
        print("MCTS Reward Output: ", MCTS_rew)
    print("Process Completed")
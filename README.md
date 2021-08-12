# UAM_scheduler
Repository for UAM Scheduling Code

## Important Information:

This program has been tested with python ver 2.7. Can check your python version by running following in terminal:

python --version 

## Versions of Packages used with Code:

numpy 1.18.5

scipy 1.5.4

matplotlib 3.3.3

## Download Instructions:

Click the green "Code" button next to the "Add File" button underneath the upper repository tool bar.

Click the "Download as Zip" option.

## main.py
These steps will walk you through how to test the MDP creation and scheduler generation process.

Step 1. Download repository.

Step 2. Open terminal and change your directory the UAM_scheduler folder.

Step 3. Run the following command in terminal: python main.py  

Step 4. Follow the prompts as they appear on the terminal.

Step 4a. The prompts will provide parameters on valid inputs. To get an idea for what values work well for the program, look at the default parameters below.

Step 5. Check the output on the terminal window after running.

# Testing main.py

### **Default Parameters for MDP Creation:** 

Enter Number of Tasks in the System (Int): 2

Do you want to use the default task configuration? (y/n): y

### **Default Parameters for Task Set:** 

Enter Computation Probability Distribution for Task 0: {1 : 1.0} 

Enter Deadline for Task 0: 2

Enter Inter-Arrival Probability Distribution for Task 0: {3: 1.0}

Enter Type of Task 0 (h/s): h

Enter Computation Probability Distribution for Task 1: {1 : 0.4, 2 : 0.5} 

Enter Deadline for Task 1: 3

Enter Inter-Arrival Probability Distribution for Task 1: {4: 1.0}

Enter Type of Task 1 (h/s): s

### **Default Parameters for MDP Creation (cont.):** 

Do you want a non-preemptible MDP? (y/n): n

Enter depth of MDP Generation (Int): 100

Enter Number of Samples (Int): 100

Enter Convergence Parameter for Value Iteration (Float): 0.01

### **Default Parameters for Value Iteration Testing:**

Enter Number of Episodes for Testing (Int): 10

### **Default Parameters for MCTS Testing:**

Would you like to run MCTS? (y/n): y

Enter Number of episodes for MCTS Testing (Int): 10

Enter Number of samples for MCTS Testing (Int): 10

Enter depth for MCTS Testing (Int): 10

Would you like to run Earliest-Deadline-First? (y/n): y

## Files:

### All of the following files are used to model the Task Generator Environment:

##### task_gen_MDP.py
This file contains the MDP class. An MDP, or Markov Decision Process is a discrete-time state transition system. 
This system is comprised of a set of states and actions. The system is used to model an environment in which an agent 
is in a state, takes an action, and moves to a next state with a fixed probability. The system follows the 
Markov property, which means that the probabilities corresponding to a specific state are independent of previous 
states. In addition, unlike Markov Chains, the agent can decide on which action to take at a specific state, 
making state transitions partly agent decision based and partly random. The MDP also contains a reward function, 
which can be used by an agent to estimate the discounted future reward for each state (also called the value of a 
state) and generate an action policy. This file is used to model a specialized instance of an MDP. The task generator 
MDP has many additional functions that are specialized for scheduler agents. These functions can be viewed in the sections below.

##### scheduler.py
This file contains the scheduler agent class. The purpose of this class is to model a scheduler traversing a task generator MDP. 
The agent is not aware of all the parameters of the MDP such as reward and probability and has to find the most optimal policy 
using an algorithm. The agent can make use of the sampling and MCTS algorithms in order to estimate probability distributions and 
estimated future reward respectively.

##### UAM_MDP.py
This file contains an alternate version of the MDP class. Just as in task_gen_MDP.py, the system is comprised of a set of states, 
a set of actions, a probability function, and a reward function. The system is used to model a further modified instance of a task
generator MDP. Unlike the task generator MDP, the UAM MDP does not allow for preemptible jobs. This means that each job that is started
by an agent has to be worked to completion before a new job can be chosen. This modification is reflected in alterations to the 
generate_MDP function that is described below.

##### scheduler_UAM.py
This file contains an alternate version of the scheduler agent class. This version of the scheduler is specialized to traverse non-preemptible
MDP environments. Just like the agent modeled in scheduler.py, the UAM scheduler can sample the MDP model and obtain an optimal policy through
value iteration and MCTS.


### task_gen_MDP:

This file contains the functions used by the MDP model class.

#### generate_MDP(depth):
This function generates a new MDP object. It does this by recursivly stepping through the task generator environment and recording 
the job configurations it reaches and the associated transition probabilities and rewards. Once the specified depth has been reached,
the function compresses the MDP into a single agent model.

#### prune_state_actions():
This function prunes the MDP of all terminal brances. It does this by recursivly stepping back from the Terminal state to all previous states.
At each previous state, the function removes all actions that lead to the terminal state. If all the actions from the previous state are removed
in this step, then the state is removed from the state set and the process is repeated.

#### value_iteration(conv_parameter, discount=0.01): 
This function uses the value iteration algorithm on the compressed MDP to calculate the optimal discounted future reward for each state. 
This discount used in the calculation process is determined by the discount parameter. The function repeats the value calculation process
until the state values converge within the conv_parameter.

#### set_policy():
After running value iteration, this function iterates over all the states in the MDP. For each state, it finds the reachable next state with 
the highest value and sets the corresponding action as the optimal action for the state. After the iteration is complete, the function returns
the optimal policy in the form of a dictionary which returns the optimal action when given a state.


### scheduler:
This file contains the functions used by the scheduler agent class when traversing a basic MDP.

#### soft_task_learning(epsilon, gamma, num_samples=0)
The soft task learning function learns the probability distributions of the task generator MDP by sampling. The algorithm can either calculate
the number of samples from the epsilon and gamma parameters, or can set the number of sampless directly with the num_samples parameter. This function 
returns the estimated probability distributions for computation time and inter-arrival time for all the tasks in the system.

#### hard_task_learning(epsilon, gamma, num_samples=0)
The soft task learning function learns the probability distributions of the task generator MDP by sampling. Before running the sampling step, 
this function runs the prune_state_actions function on the MDP environment. It then performs the sampling step on the pruned MDP, so as to 
not enter the terminal state. Just as in the soft task function, the function can calculate the number of samples from the epsilon and gamma 
parameters, or can set the number of sampless directly with the num_samples parameter. This function returns the estimated probability 
distributions for computation time and inter-arrival time for all the tasks in the system.

#### make_estimate_MDP(depth)
This function uses the probability distributions learned in the hard_task_learning or soft_task_learning functions to generate an estimate MDP. 
This function creates a new task_gen_MDP object with the estimated task set. The function then runs the generate_MDP function to set up all 
aspects of the estimated MDP

#### optimal_policy(conv_param):
This function runs value_iteration and set_policy to obtain the optimal policy of the estimated MDP.

#### test_optimal_policy(policy, num_ep=1)
This function steps through the estimated MDP, choosing an action from the provided policy dictionary at each step. The agent begins at the 
initial state of the MDP, and continues to step until it returns to the initial state or enters the terminal state. During this process, the
agent records the total reward it accrues. This process is repeated num_ep times. This function returns a dictionary with the total reward 
across each episode.

#### test_MCTS_policy(depth=10, num_samples=10, num_ep=1, rand=True):
This function is similar to test_optimal_policy, except the actions chosen at each state are provided by an MCTS algorithm. The depth parameter
indicatets to what depth the MCTS simulation should extend. The num_samples parameter corresponds to the number of simulation samples the MCTS
algorithm should take from each action. the num_ep parameter indicates the number of episodes the test function should run for. Finally the 
rand parameter indicates the simulation policy the MCTS algorithm should use. If rand = True, the simulation step will use a random policy. 
If rand = False, the simulation step will use an earliest-deadline-first policy. Just like test_optimal_policy, this function returns a dictionary
with the total reward across each episode.


### UAM_MDP:

This file contains the functions used by the non-preemptible MDP model class.

#### generate_MDP(depth):
This function generates a new MDP object. It does this by recursivly stepping through the task generator environment and recording 
the job configurations it reaches and the associated transition probabilities and rewards. Unlike the function in task_gen_MDP, 
this function generates the state space by completing each job it begins without interruption. The function does this by working on 
the same job at each time step until it completes or enters the terminal state. After the potential next states have been reached,
the function cuts out all the intermediate steps, allowing for a greatly reduced state size.

#### prune_state_actions():
This function prunes the MDP of all terminal brances. It does this by recursivly stepping back from the Terminal state to all previous states.
At each previous state, the function removes all actions that lead to the terminal state. If all the actions from the previous state are removed
in this step, then the state is removed from the state set and the process is repeated.

#### value_iteration(conv_parameter, discount=0.01): 
This function uses the value iteration algorithm on the MDP to calculate the optimal discounted future reward for each state. 
This discount used in the calculation process is determined by the discount parameter. The function repeats the value calculation process
until the state values converge within the conv_parameter.

#### set_policy():
After running value iteration, this function iterates over all the states in the MDP. For each state, it finds the reachable next state with 
the highest value and sets the corresponding action as the optimal action for the state. After the iteration is complete, the function returns
the optimal policy in the form of a dictionary which returns the optimal action when given a state.


### scheduler_UAM:
This file contains the functions used by the scheduler agent class when traversing a non-preemptible MDP.

#### soft_task_learning(epsilon, gamma, num_samples=0)
The soft task learning function learns the probability distributions of the task generator MDP by sampling. The algorithm can either calculate
the number of samples from the epsilon and gamma parameters, or can set the number of sampless directly with the num_samples parameter. This function 
returns the estimated probability distributions for computation time and inter-arrival time for all the tasks in the system.

#### hard_task_learning(epsilon, gamma, num_samples=0)
The soft task learning function learns the probability distributions of the task generator MDP by sampling. Before running the sampling step, 
this function runs the prune_state_actions function on the MDP environment. It then performs the sampling step on the pruned MDP, so as to 
not enter the terminal state. Just as in the soft task function, the function can calculate the number of samples from the epsilon and gamma 
parameters, or can set the number of sampless directly with the num_samples parameter. This function returns the estimated probability 
distributions for computation time and inter-arrival time for all the tasks in the system.

#### make_estimate_MDP(depth)
This function uses the probability distributions learned in the hard_task_learning or soft_task_learning functions to generate an estimate MDP. 
This function creates a new task_gen_MDP object with the estimated task set. The function then runs the generate_MDP function to set up all 
aspects of the estimated MDP

#### optimal_policy(conv_param):
This function runs value_iteration and set_policy to obtain the optimal policy of the estimated MDP.

#### test_optimal_policy(policy, num_ep=1)
This function steps through the estimated MDP, choosing an action from the provided policy dictionary at each step. The agent begins at the 
initial state of the MDP, and continues to step until it returns to the initial state or enters the terminal state. During this process, the
agent records the total reward it accrues. This process is repeated num_ep times. This function returns a dictionary with the total reward 
across each episode.

#### test_MCTS_policy(depth=10, num_samples=10, num_ep=1, rand=True):
This function is similar to test_optimal_policy, except the actions chosen at each state are provided by an MCTS algorithm. The depth parameter
indicatets to what depth the MCTS simulation should extend. The num_samples parameter corresponds to the number of simulation samples the MCTS
algorithm should take from each action. the num_ep parameter indicates the number of episodes the test function should run for. Finally the 
rand parameter indicates the simulation policy the MCTS algorithm should use. If rand = True, the simulation step will use a random policy. 
If rand = False, the simulation step will use an earliest-deadline-first policy. Just like test_optimal_policy, this function returns a dictionary
with the total reward across each episode.






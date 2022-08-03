import random
from task import task
import math
import numpy
from task_gen_MDP import task_gen_mdp
import time
class scheduler:
    def __init__(self, generator_object):
        self.MDP = generator_object
        self.hard_job_list = []
        self.reward_list = []
        self.totalreward = 0
        self.states = self.MDP.states
        self.e_value = 1
        self.policy = {}
        self.r = len(self.MDP.prob)


    def MCTS(self, state, depth, samples, discount = 1.0, rand=True):
        """Documentation for the MCTS method:

                This method runs the MCTS algorithm from a specified depth and returns the scheduler action with the
                maximum estimated reward. The samples parameter dictates how many samples the method takes for each
                action. The discount parameter represents the discount placed on future rewards. Finally the rand
                parameter dictates the simulation policy used by the method. If rand equals True, then the method
                will use a random simulation policy. If rand equals false, then it will use Earliest Deadline First.

        """

        cur_state = state
        out_action = None
        self.children = {}
        self.act_val = {}
        val_list = []
        for action in self.estimate_MDP.state_actions[cur_state]:
            val_reward = 0
            total_reward = 0
            for sample in range(0, samples):
                val_reward += self.MCTS_simulation_step(cur_state, action, depth, total_reward, discount = discount, rand=rand)
            self.act_val[action] = float(val_reward) / float(samples)
            val_list.append(float(val_reward) / float(samples))
        key_list = list(self.act_val.keys())
        max_val = max(val_list)
        while val_list != []:
            max_index = val_list.index(max_val)
            key = key_list[max_index]
            e_d = -1
            if key != None:
                if (state[key].d_i < e_d or e_d == -1):
                    if (0 not in state[key].c_i or state[key].c_i[0] != 1):
                        out_action = key
                        break
                    else:
                        del val_list[max_index]
                        del key_list[max_index]
                        if (len(val_list) > 0):
                            max_val = max(val_list)
                else:
                    del val_list[max_index]
                    del key_list[max_index]
                    if(len(val_list) > 0):
                        max_val = max(val_list)
            else:
                out_action = None
                break
        return out_action, max_val

    def MCTS_simulation_step(self, cur_state, action, depth, cur_reward, discount = 1.0, rand=True):
        """Documentation for the MCTS_simulation_step method:

                This method is the recursive step of the MCTS function. It begins by stepping forward from the
                cur_state by taking the given action. The method then adds the resulting reward to the cur_reward
                parameter as the total reward for the simulation step. Afterward, the performs a recursive call with
                depth-1. Once depth equals 0, the method returns the total accrued reward, multiplying by the
                discount parameter at each break-out step.

        """

        if depth == 0:
            total_reward = cur_reward
            return total_reward
        next_state, total_reward = self.estimate_MDP.next_state_scheduler_mod(tuple(cur_state), action)
        next_state_1 = self.estimate_MDP.check_in_states_s(tuple(next_state))[1]
        action_list = self.estimate_MDP.state_actions[tuple(next_state_1)]
        if rand:
            next_action = random.choice(action_list)
        else:
            if next_state_1 == tuple(["Terminal"]):
                next_action = None
            else:
                hard_deadline_dict = {}
                soft_deadline_dict = {}
                for job_index in self.estimate_MDP.state_actions[cur_state]:
                    if job_index != None:
                        if next_state_1[job_index].h_s == True and next_state_1[job_index].d_i != 0:
                            if not (0 in next_state_1[job_index].c_i and next_state_1[job_index].c_i[0] == 1.0):
                                hard_deadline_dict[job_index] = next_state_1[job_index].d_i
                        if next_state_1[job_index].h_s == False and next_state_1[job_index].d_i != 0:
                            if not (0 in next_state_1[job_index].c_i and next_state_1[job_index].c_i[0] == 1.0):
                                soft_deadline_dict[job_index] = next_state_1[job_index].d_i
                next_action = None
                if hard_deadline_dict != {}:
                    next_action = min(hard_deadline_dict, key=hard_deadline_dict.get)
                elif soft_deadline_dict != {}:
                    next_action = min(soft_deadline_dict, key=soft_deadline_dict.get)
        total_reward += self.MCTS_simulation_step(next_state_1, next_action, depth - 1, total_reward, rand=rand)
        return discount * total_reward

    def soft_task_learning(self, epsilon, gamma, num_samples=0):
        """Documentation for the soft_task_learning method:

                The soft task learning function learns the probability distributions of the task generator MDP by
                sampling. The algorithm can either calculate the number of samples from the epsilon and gamma
                parameters or can set the number of samples directly with the num_samples parameter. This function
                returns the estimated probability distributions for computation time and inter-arrival time for all
                the tasks in the system.

        """

        epsilion_1 = 1 / (math.pow(epsilon, 2) * 2)
        gamma_1 = numpy.log(2 * self.r) - numpy.log(gamma)
        self.prob_estimate_c = {}
        self.prob_estimate_a = {}
        if num_samples == 0:
            self.num_samples = self.r * math.ceil(gamma_1 * epsilion_1)
        else:
            self.num_samples = num_samples
        self.start_state = self.MDP.ret_start_state()
        for task_index in range(0, len(self.MDP.task_list)):
            self.learning_phase(task_index)
        return self.prob_estimate_c, self.prob_estimate_a

    def hard_task_learning(self, epsilon, gamma, num_samples = 0):
        """Documentation for the hard_task_learning method:

                The hard task learning function learns the probability distributions of the task generator MDP by
                sampling. Before running the sampling step, this function runs the prune_state_actions function on
                the MDP environment. It then performs the sampling step on the pruned MDP, to not enter the terminal
                state. Just as in the soft task function, the function can calculate the number of samples from the
                epsilon and gamma parameters, or can set the number of sampless directly with the num_samples
                parameter. This function returns the estimated probability distributions for computation time and
                inter-arrival time for all the tasks in the system.

        """

        self.MDP.prune_state_actions()
        epsilion_1 = 1 / (math.pow(epsilon, 2) * 2)
        gamma_1 = numpy.log(2 * self.r) - numpy.log(gamma)
        self.prob_estimate_c = {}
        self.prob_estimate_a = {}
        self.interm_estimate_c = {}
        self.interm_estimate_a = {}
        if num_samples == 0:
            self.num_samples = self.r * math.ceil(gamma_1 * epsilion_1)
        else:
            self.num_samples = num_samples
        self.start_state = self.MDP.ret_start_state()
        for task_index in range(0, len(self.MDP.task_list)):
            self.learning_phase_hard(task_index)
        return self.prob_estimate_c, self.prob_estimate_a

    def learning_phase_hard(self, task_index):
        """Documentation for the learning_phase_hard method:

                This method is the sampling step of the hard_task_learning function. The method begins by creating
                two counters, one for computation time and one for inter-arrival time. The method also creates a
                dictionary of time_step counters for computation and inter-arrival time. The method then steps
                through the MDP, taking the task_index action at each time step. In the even that the task_index
                action is not allowed in the pruned MDP, the method chooses another action from the available list.
                Every time the job at the task_index completes or is replaced by a new instance of the same job,
                the method increments the corresponding counters and counter dictionaries. After reaching the set
                number of samples, the function estimates the probability distributions by dividing the entries in
                the counter dictionaries by the total counters.

        """

        temp_counter_c = {}
        temp_counter_a = {}
        temp_counter_t = 0
        time_count = 0
        time_count_c = 0
        counter_a = 0
        counter_c = 0
        current_state = self.start_state
        prev_state = None
        start_job = self.start_state[task_index]
        next_bool = False
        finished_bool = False
        self.interm_estimate_c[task_index] = {}
        self.interm_estimate_a[task_index] = {}
        while counter_a < self.num_samples or counter_c < self.num_samples:
            if current_state != tuple(["Terminal"]) and current_state != ["Terminal"] and (
                    current_state[task_index].finish() or next_bool) and finished_bool == False:
                counter_c += 1
                finished_bool = True
                if time_count_c not in temp_counter_c:
                    temp_counter_c[time_count_c] = 1
                else:
                    temp_counter_c[time_count_c] += 1
            temp_counter_t += 1
            if current_state != tuple(["Terminal"]) and current_state != ["Terminal"] and current_state[task_index].is_same(
                    start_job) and time_count != 0:
                finished_bool = False
                if time_count not in temp_counter_a:
                    temp_counter_a[time_count] = 1
                else:
                    temp_counter_a[time_count] += 1
                time_count = 0
                time_count_c = 0
                counter_a += 1
            action = task_index
            state_1 = self.MDP.check_in_states_s(current_state)[1]
            if action not in self.MDP.state_actions[state_1]:
                action = self.MDP.state_actions[state_1][0]
            found_bool = False
            new_list = []
            new_list.extend(self.MDP.state_actions[state_1])
            while found_bool == False:
                if action != None and state_1[action].finish():
                    new_list.remove(action)
                    if new_list != []:
                        action = new_list[0]
                else:
                    found_bool = True
            if action == task_index and finished_bool == False:
                time_count_c += 1
            next_state, next_bool = self.MDP.scheduler_step_pruned(current_state, action, prev_state=prev_state, imp_index=task_index)
            prev_state = current_state
            current_state = next_state
            if counter_c != 0 and counter_c % 1000 == 0:
                self.interm_estimate_c[task_index][counter_c] = {}
                for key in temp_counter_c.keys():
                    self.interm_estimate_c[task_index][counter_c][key] = float(temp_counter_c[key]) / float(counter_c)
            if counter_a != 0 and counter_a % 1000 == 0:
                self.interm_estimate_a[task_index][counter_a] = {}
                for key in temp_counter_a.keys():
                    self.interm_estimate_a[task_index][counter_a][key] = float(temp_counter_a[key]) / float(counter_a)
            time_count += 1

        self.prob_estimate_c[task_index] = {}
        for key in temp_counter_c.keys():
            self.prob_estimate_c[task_index][key] = temp_counter_c[key] / float(counter_c)

        self.prob_estimate_a[task_index] = {}
        for key in temp_counter_a.keys():
            self.prob_estimate_a[task_index][key] = temp_counter_a[key] / float(counter_a)

    def learning_phase(self, task_index):
        """Documentation for the learning_phase method:

                This method is the sampling step of the hard_task_learning function. The method begins by creating
                two counters, one for computation time and one for inter-arrival time. The method also creates a
                dictionary of time_step counters for computation and inter-arrival time. The method then steps
                through the MDP, taking the task_index action at each time step. Every time the job at the
                task_index completes or is replaced by a new instance of the same job, the method increments the
                corresponding counters and counter dictionaries. After reaching the set number of samples,
                the function estimates the probability distributions by dividing the entries in the counter
                dictionaries by the total counters.

        """

        temp_counter_c = {}
        temp_counter_a = {}
        temp_counter_t = 0
        time_count = 0
        counter_a = 0
        counter_c = 0
        current_state = self.start_state
        start_job = self.start_state[task_index]
        next_bool = False
        finished_bool = False
        while counter_a < self.num_samples or counter_c < self.num_samples:
            if current_state!= tuple(["Terminal"]) and (current_state[task_index].finish() or next_bool) and finished_bool == False:
                counter_c += 1
                finished_bool = True
                if time_count not in temp_counter_c:
                    temp_counter_c[time_count] = 1
                else:
                    temp_counter_c[time_count] += 1
            temp_counter_t += 1
            if current_state != tuple(["Terminal"]) and current_state[task_index].is_same(start_job) and time_count != 0:
                finished_bool = False
                if time_count not in temp_counter_a:
                    temp_counter_a[time_count] = 1
                else:
                    temp_counter_a[time_count] += 1
                time_count = 0
                counter_a += 1
            next_state, next_bool = self.MDP.scheduler_step(current_state, task_index)
            current_state = next_state
            time_count += 1

        self.prob_estimate_c[task_index] = {}
        for key in temp_counter_c.keys():
            self.prob_estimate_c[task_index][key] = temp_counter_c[key] / float(counter_c)

        self.prob_estimate_a[task_index] = {}
        for key in temp_counter_a.keys():
            self.prob_estimate_a[task_index][key] = temp_counter_a[key] / float(counter_a)
        return [self.prob_estimate_c, self.prob_estimate_a]

    def make_estimate_MDP(self, depth):
        """Documentation for the make_estimate_MDP method:

                This function uses the probability distributions learned in the hard_task_learning or
                soft_task_learning functions to generate an estimate MDP. This function creates a new task_gen_MDP
                object with the estimated task set. The function then runs the generate_MDP function to set up all
                aspects of the estimated MDP

        """

        new_task_list = []
        for task_index in self.prob_estimate_a:
            new_task_list.append(task(self.prob_estimate_c[task_index], self.MDP.task_list[task_index].d_i, self.prob_estimate_a[task_index], hard=self.MDP.task_list[task_index].hard))
        self.estimate_MDP = task_gen_mdp(new_task_list)
        self.estimate_MDP.generate_MDP(depth)

        return self.estimate_MDP

    def optimal_policy(self, conv_param):
        """Documentation for the optimal_policy method:

                This function runs value_iteration and set_policy to obtain the optimal policy of the estimated MDP.

        """
        tic = time.time()
        self.estimate_MDP.value_iteration(conv_param)
        toc = time.time()
        time_val = toc - tic
        out_pol = self.estimate_MDP.set_policy()
        return out_pol, time_val

    def test_optimal_policy(self, policy, num_ep=1):
        """Documentation for the test_optimal_policy method:

                This function steps through the estimated MDP, choosing an action from the provided policy dictionary
                at each step. The agent begins at the initial state of the MDP, and continues to step until it
                returns to the initial state or enters the terminal state. During this process, the agent records the
                total reward it accrues. This process is repeated num_ep times. This function returns a dictionary
                with the total reward across each episode.

        """

        reward_dict = {}
        reward = 0
        for episode in range(1, 1 + num_ep):
            finished_bool = False
            temp_state = self.estimate_MDP.ret_start_state()
            start_state = self.estimate_MDP.check_in_states_s(temp_state)[1]
            current_state = self.estimate_MDP.check_in_states_s(temp_state)[1]
            while not finished_bool:
                action = policy[current_state]
                temp_state, next_bool = self.estimate_MDP.scheduler_step(current_state, action)
                next_state = self.estimate_MDP.check_in_states_s(temp_state)[1]
                reward += self.estimate_MDP.reward_mod(current_state, action, next_state)
                current_state = next_state
                if current_state == tuple(["Terminal"]):
                    finished_bool = True
                if self.estimate_MDP.check_if_equal(current_state, start_state) == True:
                    finished_bool = True
            reward_dict[episode] = reward

        return reward_dict

    def test_MCTS_policy(self, depth=10, num_samples=10, num_ep=1, rand=True):
        """Documentation for the test_MCTS_policy method:

                This function is similar to test_optimal_policy, except the actions chosen at each state are provided
                by an MCTS algorithm. The depth parameter indicates to what depth the MCTS simulation should extend.
                The num_samples parameter corresponds to the number of simulation samples the MCTS algorithm should
                take from each action. the num_ep parameter indicates the number of episodes the test function should
                run for. Finally, the rand parameter indicates the simulation policy the MCTS algorithm should use.
                If rand = True, the simulation step will use a random policy. If rand = False, the simulation step
                will use an earliest-deadline-first policy. Just like test_optimal_policy, this function returns a
                dictionary with the total reward across each episode.

        """
        reward_dict = {}
        reward = 0
        time_counter = 0
        time_total = 0
        time_list = []
        for episode in range(1, 1 + num_ep):
            finished_bool = False
            temp_state = self.estimate_MDP.ret_start_state()
            start_state = self.estimate_MDP.check_in_states_s(temp_state)[1]
            current_state = self.estimate_MDP.check_in_states_s(temp_state)[1]
            while not finished_bool:
                time_counter += 1
                time_count_MCTS = time.time()
                action, out_val = self.MCTS(current_state, depth, num_samples, rand=rand)
                time_count_end = time.time()
                time_list.append(time_count_end - time_count_MCTS)
                time_total += time_count_end - time_count_MCTS
                temp_state, next_bool = self.estimate_MDP.scheduler_step(current_state, action)
                next_state = self.estimate_MDP.check_in_states_s(temp_state)[1]
                reward += self.estimate_MDP.reward_mod(current_state, action, next_state)
                current_state = next_state
                if current_state == tuple(["Terminal"]):
                    finished_bool = True
                if self.estimate_MDP.check_if_equal(current_state, start_state) == True:
                    finished_bool = True
            reward_dict[episode] = reward
        return reward_dict


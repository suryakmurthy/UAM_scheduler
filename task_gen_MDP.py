import numpy as np
import numpy
import itertools
from task_generator import task_generator
from job import job

class task_gen_mdp:
    def __init__(self, task_list):
        """Documentation for the __init__ method:

                This method initializes the uam_mdp object. The function takes a list of task objects as an argument.
                The function creates a task_generator object based on the task list and an initial state from the job
                instances generated by the task objects.

        """

        self.job_list = []
        self.task_list = task_list
        self.t_g = task_generator(self.task_list)
        self.q_table = dict([])
        self.convergence = 0
        for task_1 in task_list:
            self.job_list.append(task_1.generate_job())

    def next_state_scheduler(self, state, action_index):
        """Documentation for the next_state_scheduler method:

                This method uses the known rules of the scheduler agent to generate a next state object given a
                current state and action. The action_index parameter corresponds to the index of the job being worked
                on during this time step. This function creates a new job object for each job in the state,
                and steps forward. If the index of the job within the state matches the action index, then the all of
                the jobs time-based parameters are decremented (with a minimum of 0). For all other jobs,
                all the time-based parameters except the computation time are decremented. This is handled by the
                built-in step function of the job class.
        """
        next_state = []
        if action_index in self.action_task_full:
            action_index = None
        for job_index in range(0, len(state)):
            if state[job_index] == "Terminal":
                next_state = ["Terminal"]
                break
            new_job = job(state[job_index].c_i, state[job_index].d_i, state[job_index].a_i, hard=state[job_index].h_s)
            if action_index == job_index:
                new_job.step(True)
            else:
                new_job.step(False)
            if new_job.d_i == 0 and new_job.h_s == False:
                if 0 not in new_job.c_i or new_job.c_i[0] != 1.0:
                    new_job.past_deadline = True
            next_state.append(new_job)
        return next_state

    def next_state_task(self, state, action_list):
        """Documentation for the next_state_task method:

                This method uses the known rules of the task generator agent to generate a next state object given a
                current state and action. The action_list parameter contains a list of task_generator actions. Each
                action within this list corresponds to a job in the state list. If the action equals 'sub' or
                'killANDsub', the method replaces the job with a new job object generated from the corresponding task
                in the task list. If the action 'fin', the method replaces the computation time parameter with {0 :
                1.0} to signify that the job has finished computation. If the action equals 'e' and no time-based
                parameters equal 0, then the task generator does nothing to the job. If the action equals 'e' any
                computation or inter-arrival entries are 0, the function removes the entry and merges the remaining
                probabilities.
        """
        next_state = []
        bool_check = False
        if action_list not in self.action_task_full:
            action_list = ['e' for element in range(0, len(self.task_list))]
        for job_index in range(0, len(state)):
            action = action_list[job_index]
            if state[job_index] == "Terminal":
                next_state = ["Terminal"]
                break
            new_job = job(state[job_index].c_i, state[job_index].d_i, state[job_index].a_i, hard=state[job_index].h_s)
            if state[job_index].d_i == 0 and state[job_index].h_s == True and action != "fin":
                if 0 not in state[job_index].c_i.keys() or state[job_index].c_i[0] != 1.0:
                    next_state = ["Terminal"]
                    break
            if action == "sub" or action == "killANDsub":
                new_job = self.task_list[job_index].generate_job()
            if action == "fin":
                new_job.c_i = {0 : 1.0}
                new_job.finished = True
            if action != "sub" and action != "killANDsub":
                if 0 in new_job.c_i.keys() and new_job.c_i[0] != 1:
                    new_c_i = {}
                    non_zero_prob =  1 - new_job.c_i[0]
                    for key in new_job.c_i.keys():
                        if key != 0:
                            TOLERANCE = 0.00000001
                            LOW = 1 - TOLERANCE
                            HIGH = 1 + TOLERANCE
                            c_i_temp = new_job.c_i[key] / non_zero_prob
                            if c_i_temp < HIGH and c_i_temp > LOW:
                                new_c_i[key] = 1.0
                            else:
                                new_c_i[key] = new_job.c_i[key] / non_zero_prob
                    new_job.c_i = new_c_i
                if 0 in new_job.a_i.keys() and new_job.a_i[0] != 1:
                    new_a_i = {}
                    non_zero_prob = 1 - new_job.a_i[0]
                    for key in new_job.a_i.keys():
                        if key != 0:
                            TOLERANCE = 0.00000001
                            LOW = 1 - TOLERANCE
                            HIGH = 1 + TOLERANCE
                            a_i_temp = new_job.a_i[key] / non_zero_prob
                            if a_i_temp < HIGH and a_i_temp > LOW:
                                new_a_i[key] = 1.0
                            else:
                                new_a_i[key] = new_job.a_i[key] / non_zero_prob
                    if new_a_i == {}:
                        new_a_i[0] = 1.0
                    new_job.a_i = new_a_i
            new_job.past_deadline = state[job_index].past_deadline
            if new_job.d_i == 0 and new_job.h_s == False:
                if 0 not in new_job.c_i or new_job.c_i[0] != 1.0:
                    new_job.past_deadline = True
            next_state.append(new_job)
        return next_state

    def generate_MDP(self, depth):
        """Documentation for the generate_MDP method:

                This function generates a new MDP object. It does this by recursively stepping through the task
                generator environment and recording the job configurations it reaches and the associated transition
                probabilities and rewards. Once the specified depth has been reached, the function compresses the MDP
                into a single agent model.

        """

        self.check_bool = False
        self.state_list_scheduler = [tuple(self.job_list)]
        self.state_list_task = []
        self.states = [tuple(self.job_list)]
        self.prob = {}
        self.prev_list = {}
        self.state_actions = {}
        self.actions_scheduler = list(range(0, len(self.job_list)))
        self.actions_task = ["sub", "fin", "killANDsub", "e"]
        list_1 = []
        for task in self.task_list:
            list_1.append(self.actions_task)
        self.action_task_full = list(itertools.product(*list_1))
        self.actions_scheduler.append(None)
        self.state_actions[tuple(self.job_list)] = list(self.actions_scheduler)
        self.reward = {}
        self.next_state_dict_s = {}
        self.next_state_dict_t = {}
        self.prior_state_dict = {tuple(self.job_list): []}
        self.actions = self.action_task_full + self.actions_scheduler
        self.generate_rec(depth - 1, self.job_list)
        self.make_mod_MDP()

    def generate_rec(self, depth, cur_state):
        """Documentation for the generate_rec method:

                The generate rec method represents the recursive step of the generate_MDP function. This function
                begins at a given starting state, and obtains a next state for each scheduler action by using the
                next_state_scheduler function. For each next state, the method steps forward again using the
                next_state_task function. After this step, the method adds a entry to the probability and reward
                dictionaries for each next state. If a next state have already been visited, the system does not add
                it to the set of next states for the next recursive step. For each next state, the system recursively
                calls itself with a depth - 1. Once the depth equals 0, the function breaks out of the recursive step.
        """

        if depth == 0:
            return
        scheduler_actions = list(range(0, len(cur_state)))
        scheduler_actions.append(None)
        next_states = []
        for action in self.actions_scheduler:
            next_state = self.next_state_scheduler(cur_state, action)
            in_states_s = self.check_in_states_t(tuple(next_state))
            if in_states_s[0] == True:
                next_state = list(in_states_s[1])
                if next_state != ["Terminal"]:
                    for job in next_state:
                        if 0 in job.c_i.keys():
                            next_states.append(next_state)
                            break
            if in_states_s[0] == False:
                self.state_list_task.append(tuple(next_state))
                next_states.append(next_state)
                if next_state != ["Terminal"]:
                    self.states.append(tuple(next_state))
                if next_state == ["Terminal"] and not self.check_in_states(tuple(next_state)):
                    self.states.append(tuple(next_state))
            task_actions = self.t_g.ret_possible_actions(next_state, prev_state=cur_state)
            if tuple(next_state) in self.state_actions:
                for action_1 in task_actions:
                    if action_1 not in self.state_actions[tuple(next_state)]:
                        self.state_actions[tuple(next_state)].append(action_1)
            else:
                self.state_actions[tuple(next_state)] = task_actions
            self.next_state_dict_s[(tuple(cur_state), action)] = tuple(next_state)
            self.prob[(tuple(cur_state), action, tuple(next_state))] = 1
            if tuple(next_state) not in self.prior_state_dict:
                self.prior_state_dict[(tuple(next_state))] = []
            self.prior_state_dict[(tuple(next_state))].append((tuple(cur_state), action))
            self.reward[(tuple(cur_state), action, tuple(next_state))] = 0
            for element in next_state:
                if element == "Terminal" or element.failed:
                    self.reward[(tuple(cur_state), action, tuple(next_state))] += -10000000
                    break
                if element.d_i == 0 and ((0 not in element.c_i) or (element.c_i[0] != 1.0)) and element.h_s == False:
                    self.reward[(tuple(cur_state), action, tuple(next_state))] = -10
        next_scheduler_states = []
        for state in next_states:
            task_actions = self.t_g.ret_possible_actions(state, prev_state=cur_state)
            for action in task_actions:
                next_state = self.next_state_task(state, action)
                in_states_t = self.check_in_states_s(tuple(next_state))
                if in_states_t[0] == True:
                    next_state = list(in_states_t[1])
                if in_states_t[0] == False:
                    self.state_list_scheduler.append(tuple(next_state))
                    next_scheduler_states.append(next_state)
                    self.states.append(tuple(next_state))
                self.prob[(tuple(state), tuple(action), tuple(next_state))] = 1
                self.next_state_dict_t[(tuple(state), tuple(action))] = tuple(next_state)
                self.state_actions[tuple(next_state)] = list(self.actions_scheduler)
                if tuple(next_state) not in self.prior_state_dict:
                    self.prior_state_dict[(tuple(next_state))] = []
                self.prior_state_dict[(tuple(next_state))].append((tuple(state), action))
                self.reward[(tuple(state), action, tuple(next_state))] = 0
                for element in next_state:
                    if element == "Terminal" or element.failed:
                        self.reward[(tuple(state), action, tuple(next_state))] += -10000000
                        break
                    if element.d_i == 0 and ((0 not in element.c_i) or (element.c_i[0] != 1.0)) and element.h_s == False:
                        self.reward[(tuple(state), action, tuple(next_state))] += -10
        for next_s_s in next_scheduler_states:
            self.generate_rec(depth - 1, next_s_s)

    def ret_start_state(self):
        """Documentation for the ret_start_state method:

                Returns the starting state for the MDP system. The starting state is defined as the state containing
                the initial job configurations for each task in the system.
        """

        output = []
        for task_1 in self.task_list:
            output.append(task_1.generate_job())
        out_1 = tuple(output)
        out_2 = self.check_in_states(output)
        return out_2[1]

    def possible_prior(self, state):
        """Documentation for the possible_prior method:

               Returns the prior states for a given input state
        """

        prior_states = self.prior_state_dict[state]
        return prior_states

    def prune_state_actions(self):
        """Documentation for the prune_state_actions method:

                This function prunes the MDP of all terminal branches. It does this by recursively stepping back from
                the Terminal state to all previous states. At each previous state, the function removes all actions
                that lead to the terminal state. If all the actions from the previous state are removed in this step,
                then the state is removed from the state set and the process is repeated.

        """

        self.new_states = self.states
        R = [tuple(["Terminal"])]
        T = [tuple(["Terminal"])]
        while R != []:
            t = R[0]
            new_R = list(R)
            new_R.remove(t)
            prior_states = self.possible_prior(t)
            for (state, action) in prior_states:
                if state != t:
                    if action in self.state_actions[state]:
                        self.state_actions[state].remove(action)
                    if self.state_actions[state] == []:
                        new_R.append(state)
                        T.append(state)
            R = new_R

        self.new_states = [i for i in self.states if i not in T]

    def possible_next(self, state, action_list=[], scheduler_bool=True):
        """Documentation for the possible_next method:

                Returns the next states for a given input state and list of actions. The scheduler bool parameter is
                True if the current state is a scheduler state and false if it is a task generator state.
        """
        next_states = []
        bool_t = self.check_in_states_s(state)
        bool_check = False
        if scheduler_bool:
            next_states.append(tuple(self.next_state_dict_s[(tuple(state), action_list[0])]))
        else:
            next_states.append(tuple(self.next_state_dict_t[(tuple(state), action_list[0])]))
        return next_states

    def possible_next_pruned(self, state, action):
        """Documentation for the possible_next_pruned method:

                Returns the next states for a given input state and list of actions assuming a pruned MDP.
        """
        next_states = []
        next_rewards = {}
        bool_t = self.check_in_states_s(state)
        next_state_1 = tuple(self.next_state_dict_s[(tuple(state), action)])
        next_states_2 = []
        for action in self.state_actions[next_state_1]:
            next_state_3 = self.check_in_states_s(self.next_state_task(next_state_1, action))[1]
            next_states_2.append(next_state_3)
            next_rewards[next_states_2] = self.get_reward(next_state_1, action, next_state_3)
        return next_states_2, next_rewards

    def obtain_chain(self, starting_state, action_set):
        """Documentation for the obtain_chain method:

                This method takes a starting_state and list of actions as arguments. The method begins at the
                starting state and iterates over all the actions in the list. For each action, the method moves to a
                next state using the next_state_scheduler function. The method outputs a list of the next states it
                reached during iteration.
        """
        current_state = starting_state
        state_list = []
        for action_index in range(0, len(action_set)):
            if action_index % 2 == 0:
                next_state = self.next_state_scheduler(current_state, action_set[action_index])
            else:
                next_state = self.next_state_task(current_state, action_set[action_index])
            state_list.append(next_state)
            current_state = next_state
        return state_list

    def check_if_equal(self, state_1, state_2):
        """Documentation for the check_if_equal method:

                Returns True if the two argument states contain the same job objects. If the two arguments aren't
                equal, the method returns False.
        """
        equal_bool = True
        if state_1 == tuple(["Terminal"]) and state_2 == tuple(["Terminal"]):
            return True
        if (state_1 == tuple(["Terminal"]) and state_2 != tuple(["Terminal"])) or (state_1 != tuple(["Terminal"]) and state_2 == tuple(["Terminal"])):
            return False
        for job_index in range(0, len(state_1)):
            job_1 = state_1[job_index]
            job_2 = state_2[job_index]
            if not job_1.is_same(job_2):
                equal_bool = False
        return equal_bool

    def scheduler_step(self, state, action, rew=False):
        """Documentation for the scheduler_step method:

                Returns a next state given a state and action. The method also returns True if the job begin worked
                on completes during the step. If the rew argument is True, the function also returns the total
                reward over the step.
        """
        if state == tuple(["Terminal"]):
            if rew:
                return state, False, self.get_reward(state, None, tuple(["Terminal"]))
            else:
                return state, False
        total_reward = 0
        bool_1 = self.check_in_states_s(state)
        if not bool_1[0]:
            return
        state = bool_1[1]
        possible_next = self.possible_next(state, action_list=[action])
        prob_dict = {}
        for next_state_index in range(0, len(possible_next)):
            prob_dict[next_state_index] = self.transition(state, action, possible_next[next_state_index], transition=True)
        next_state = possible_next[np.random.choice(prob_dict.keys(), p=prob_dict.values())]
        total_reward += self.get_reward(state, action, next_state, transition=True)

        if action !=  None:
            if next_state[action].finish():
                out_bool = True
            else:
                out_bool = False
        else:
            out_bool = False
        if rew == True:
            final_state, temp_reward = self.task_gen_step(next_state, rew_1=rew)
            total_reward += temp_reward
        else:
            final_state = self.task_gen_step(next_state, rew_1=rew)
        if rew == False:
            return final_state, out_bool
        else:
            return final_state, out_bool, total_reward

    def scheduler_step_pruned(self, state, action, prev_state = None, imp_index=-1):
        """Documentation for the scheduler_step_pruned method:

                Returns a next state given a state and action. Works similarly to the scheduler_step method,
                but has some exception cases to handle the pruned MDP.

        """
        if imp_index == -1:
            imp_index_1 = action
        else:
            imp_index_1 = imp_index
        bool_1 = self.check_in_states_s(state)
        if not bool_1[0]:
            return
        state = bool_1[1]
        if action not in self.state_actions[state]:
            action = self.state_actions[state][0]
        possible_next = self.possible_next(state, action_list=[action])
        prob_dict = {}
        for next_state_index in range(0, len(possible_next)):
            prob_dict[next_state_index] = self.transition(state, action, possible_next[next_state_index], transition=True)
        next_state = possible_next[np.random.choice(prob_dict.keys(), p=prob_dict.values())]
        if next_state[imp_index_1].finish():
            out_bool = True
        else:
            out_bool = False
        final_state = self.task_gen_step_pruned(next_state, prev_state=state)
        return final_state, out_bool

    def task_gen_step_pruned(self, state, prev_state = None):
        """Documentation for the task_gen_step_pruned method:

                Returns a next state given a state and action. Works similarly to the task_gen_step method,
                but has some exception cases to handle the pruned MDP.
        """
        action_list = self.t_g.ret_possible_actions(state, prev_state=prev_state)
        if (all(action in self.state_actions[state] for action in action_list)):
            action_1 = tuple(self.t_g.ret_action(state))
        else:
            action_1 = self.state_actions[state][0]
        next_state = self.next_state_task(state, action_1)
        return next_state

    def task_gen_step(self, state, rew_1=False):
        """Documentation for the task_gen_step method:

                Returns a next scheduler state given a state.
        """
        action_1 = tuple(self.t_g.ret_action(state))
        next_state = self.next_state_task(state, action_1)
        bool_1 = self.check_in_states_s(tuple(next_state))
        if bool_1[0]:
            reward = self.get_reward(state, action_1, bool_1[1], transition=False)
        else:
            reward = 0
        if rew_1:
            return tuple(next_state), reward
        else:
            return tuple(next_state)

    def check_in_states_s(self, state):
        """Documentation for the check_in_states_s method:

                Checks if the input state is in the set of scheduler states. If so, the method returns True. If not,
                it returns False.

        """
        if self.state_list_scheduler == []:
            return (False, None)
        if state == tuple(["Terminal"]):
            if state in self.state_list_scheduler:
                return (True, tuple(["Terminal"]))
            else:
                return (False, None)
        for other_state in self.state_list_scheduler:
            if self.check_if_equal(state, other_state):
                return (True, other_state)
        return (False, None)

    def check_in_states_t(self, state):
        """Documentation for the check_in_states_t method:

                Checks if the input state is in the set of task generator states. If so, the method returns True. If
                not, it returns False.

        """
        if self.state_list_task == []:
            return (False, None)
        if state == tuple(["Terminal"]):
            if state in self.state_list_task:
                return (True, tuple(["Terminal"]))
            else:
                return (False, None)
        for other_state in self.state_list_task:
            if self.check_if_equal(state, other_state):
                return (True, other_state)
        return (False, None)

    def check_in_states(self, state):
        """Documentation for the check_in_states method:

                Checks if the input state is in the set of all states. If so, the method returns True. If
                not, it returns False.

        """
        if self.states == []:
            return (False, None)
        if state == tuple(["Terminal"]):
            if state in self.states:
                return (True, tuple(["Terminal"]))
            else:
                return (False, None)
        for other_state in self.states:
            if self.check_if_equal(state, other_state):
                return (True, other_state)
        return (False, None)

    def transition(self, state, action, next_state, transition=True):
        """Documentation for the transition method:

                Checks if the input state, action, next_state tuple is probability dictionary. If so, the method
                returns the probability entry. If not, the method returns 0.

        """
        if transition:
            prob_1 = self.check_in_states_s(state)
            prob_2 = self.check_in_states_t(next_state)
        else:
            prob_1 = self.check_in_states_t(state)
            prob_2 = self.check_in_states_s(next_state)
        if prob_1[0] and prob_2[0]:
            if (prob_1[1], action, prob_2[1]) in self.prob.keys():
                return self.prob[(prob_1[1], action, prob_2[1])]
            else:
                return 0
        else:
            return 0

    def get_reward(self, state, action, next_state, transition=True):
        """Documentation for the get_reward method:

                Checks if the input state, action, next_state tuple is reward dictionary. If so, the method
                returns the reward entry. If not, the method returns 0.

        """
        if transition:
            prob_1 = self.check_in_states_s(state)
            prob_2 = self.check_in_states_t(next_state)
        else:
            prob_1 = self.check_in_states_t(state)
            prob_2 = self.check_in_states_s(next_state)
        if prob_1[0] and prob_2[0]:
            if (prob_1[1], action, prob_2[1]) in self.reward.keys():
                return self.reward[(prob_1[1], action, prob_2[1])]
            else:
                return 0
        else:
            return 0

    def value_iteration(self, conv_parameter, discount=0.01):
        """Documentation for the value_iteration method:

            This method is a loop for the value calculation method. It takes a k parameter and iterates the
            value_calculation method k times.
        """
        self.discount = discount
        self.value = numpy.zeros(len(self.state_list_scheduler))
        self.convergence = conv_parameter
        self.finish_iteration = False
        while not self.finish_iteration:
            self.value_calculation()

    def value_calculation(self):
        """Documentation for the value_calculation method:

            The value_calculation calculates the discounted future reward at each state for each iteration. The
            explanation behind the value calculation algorithm can be found at this link:
            https://artint.info/2e/html/ArtInt2e.Ch9.S5.html. The program begins by calculating the discounted future
            reward at a state given an action (q-value), then taking the maximum q-value as the value of the state.
            It continues this calculation over all states in the state set.
        """
        value_k = self.value.copy()
        count = 0
        for state in self.state_list_scheduler:
            if self.check_in_states_s(state)[0]:
                t = True
            else:
                t = False
            count += 1
            total = [None] * len(self.state_actions[state])
            for index_1 in range(0, len(self.state_actions[state])):
                action = self.state_actions[state][index_1]
                total[index_1] = 0
                for next_state in self.state_list_scheduler:
                    add_prob = self.transition_mod(state, action, next_state)
                    add_reward = self.reward_mod(state, action, next_state)
                    add_disc = self.discount * self.value[self.state_list_scheduler.index(next_state)]
                    add_total = (add_prob * ((add_reward) + (add_disc)))
                    total[index_1] += add_total
                self.q_table[(state, action)] = total[index_1]

            value_k[self.state_list_scheduler.index(state)] = max(total)
        finish_counter = 0
        for index in range(self.value.size):
            if abs(self.value[index] - value_k[index]) <= self.convergence:
                finish_counter += 1
        if finish_counter >= self.value.size - 1:
            self.finish_iteration = True
        self.value = value_k.copy()

    def make_mod_MDP(self):
        """Documentation for the make_mod_MDP method:

                This function compresses the given MDP by removing the task_generator transition probabilities and
                causing the MDP to be comprised of only scheduler states. This allows the MDP model to run Value
                Iteration.
        """
        self.new_transition = {}
        self.new_reward = {}
        for state in self.state_list_scheduler:
            for action in self.state_actions[state]:
                next_state, next_rew = self.possible_next_scheduler(state, action)
                for state_2 in next_state.keys():
                    self.new_transition[(state, action, state_2)] = next_state[state_2]
                    self.new_reward[(state, action, state_2)] = next_rew[state_2]

    def possible_next_scheduler(self, state, action):
        """Documentation for the possible_next_scheduler method:

                This function returns all of the possible next scheduler states given a scheduler state and an
                action. The function uses the possible_next method to obtain the task_generator state from the input
                state. After this step, the function uses obtains the task generator action distribution using the
                ret_action_dist function. By finding all of the next states from the task generator state,
                the function returns a probability dictionary and a reward dictionary for the next states.
        """

        next_1 = self.possible_next(state, action_list=[action])[0]
        list_1 = self.t_g.ret_action_dist(list(next_1), prev_state=state)
        rew_1 = self.get_reward(state, action, next_1)
        out_dict = {}
        rew_dict = {}
        for action_2 in list_1.keys():
            next_2 = self.possible_next(next_1, action_list=[action_2], scheduler_bool=False)[0]
            next_3 = self.check_in_states_s(next_2)[1]
            rew_dict[next_3] = self.get_reward(next_1, action_2, next_3, transition=False) + rew_1
            if next_3 not in out_dict:
                out_dict[next_3] = list_1[action_2]
            else:
                out_dict[next_3] += list_1[action_2]
        return out_dict, rew_dict

    def transition_mod(self, state, action, next_state):
        """Documentation for the transition_mod method:

              This function is a transition function for the compressed MDP. Its function is the same as the
              transition function, but it only works if state and next_state are both scheduler states.
        """
        if (state, action, next_state) in self.new_transition:
            return self.new_transition[(state, action, next_state)]
        else:
            return 0

    def reward_mod(self, state, action, next_state):
        """Documentation for the reward_mod method:

              This function is a reward function for the compressed MDP. Its function is the same as the
              reward function, but it only works if state and next_state are both scheduler states.
        """

        if (state, action, next_state) in self.new_reward:
            return self.new_reward[(state, action, next_state)]
        else:
            return 0

    def next_state_scheduler_mod(self, state, action):
        """Documentation for the next_state_scheduler_mod method:

              This function is a next state function for the compressed MDP. Its function is the same as the
              next_state_scheduler function, but returns a probability distribution dict of scheduler states.
        """

        dist_dict = {}
        for next_state_index in range(0, len(self.state_list_scheduler)):
            next_state = self.state_list_scheduler[next_state_index]
            prob = self.transition_mod(state, action, next_state)
            if prob != 0:
                dist_dict[next_state_index] = prob
        true_next_index = np.random.choice(dist_dict.keys(), p=dist_dict.values())
        true_next = self.state_list_scheduler[true_next_index]
        reward = self.reward_mod(state, action, true_next)
        return true_next, reward

    def set_policy(self):
        """Documentation for the set_policy method:

             This method uses the q-value dictionary to generate a policy for the MDP. The q-values represent the
             discounted future reward at a state if you take a specific action, and then act optimally. This method
             find the maximum q-value for each state and uses the corresponding action as the optimal action at the
             given state. The method then stores this action in the correct index in the policy list.
        """
        policy_1 = {}
        for state in self.state_list_scheduler:
            if self.check_in_states_s(state)[0]:
                t = True
            else:
                t = False
            action_values = [None] * len(self.state_actions[state])
            for action in self.state_actions[state]:
                index_1 = self.state_actions[state].index(action)
                action_values[index_1] = 0
                for next_state in self.state_list_scheduler:
                    action_values[index_1] += (self.transition_mod(state, action, next_state) * (
                            self.reward_mod(state, action, next_state) + (
                            self.discount * self.value[self.state_list_scheduler.index(next_state)])))
            policy_1[state] = self.state_actions[state][action_values.index(max(action_values))]
        return policy_1

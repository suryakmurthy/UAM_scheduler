import numpy as np
import numpy
import itertools
from task_generator import task_generator
from job import job

class uam_mdp:
    def __init__(self, task_list):
        self.job_list = []
        self.task_list = task_list
        self.t_g = task_generator(self.task_list)
        self.q_table = dict([])
        self.discount = 0.01
        self.convergence = 0
        for task_1 in task_list:
            self.job_list.append(task_1.generate_job())

    def next_state_scheduler(self, state, action_index):
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
        self.check_bool = False
        self.state_list_scheduler = [tuple(self.job_list)]
        self.state_list_task = []
        self.states = [tuple(self.job_list)]
        self.prob = {}
        self.time_steps = {}
        self.prev_list = {}
        self.state_actions = {}
        self.actions_scheduler = list(range(0, len(self.job_list)))
        list_1 = []
        self.actions_task = ["sub", "fin", "killANDsub", "e"]
        for task in self.task_list:
            list_1.append(self.actions_task)
        self.action_task_full = list(itertools.product(*list_1))
        self.actions_scheduler.append(None)
        self.state_actions[tuple(self.job_list)] = list(self.actions_scheduler)
        self.reward = {}
        self.next_state_dict_s = {}
        self.next_state_dict_t = {}
        self.prior_state_dict = {tuple(self.job_list): []}
        self.actions = self.actions_scheduler
        self.generate_rec(depth - 1, self.job_list)

    def generate_rec(self, depth, start_state):
        if depth == 0:
            return
        scheduler_actions = list(range(0, len(start_state)))
        scheduler_actions.append(None)
        next_states = []
        self.state_actions[tuple(start_state)] = list(range(0, len(self.job_list)))
        next_state_list = []
        for action in self.actions_scheduler:
            self.total_prob = {tuple(start_state): 1.0}
            self.total_reward = {tuple(start_state): 0.0}
            self.sub_time_steps = {tuple(start_state): 0}
            self.sub_prev_states = {}
            self.sub_step_states = []
            log_bool = False
            new_state_list = self.rec_sub_step(start_state, action, log_bool=log_bool)
            for next_state in new_state_list:
                in_states_t = self.check_in_states_s(tuple(next_state))
                prob_1 = self.total_prob[tuple(next_state)]
                time_1 = self.sub_time_steps[tuple(next_state)]
                reward_1 = self.total_reward[tuple(next_state)]
                if in_states_t[0] == True:
                    temp_state = next_state
                    next_state = list(in_states_t[1])
                if in_states_t[0] == False:
                    self.state_list_scheduler.append(tuple(next_state))
                    next_states.append(next_state)
                    self.states.append(tuple(next_state))
                if (tuple(start_state), action, tuple(next_state)) not in self.prob:
                    self.prob[(tuple(start_state), action, tuple(next_state))] = prob_1
                if (tuple(start_state), action, tuple(next_state)) not in self.reward:
                    self.reward[(tuple(start_state), action, tuple(next_state))] = reward_1
                if (tuple(start_state), action) not in self.next_state_dict_s.keys():
                    self.next_state_dict_s[(tuple(start_state), action)] = [tuple(next_state)]
                else:
                    if tuple(next_state) not in self.next_state_dict_s[(tuple(start_state), action)]:
                        self.next_state_dict_s[(tuple(start_state), action)].append(tuple(next_state))
                if (tuple(next_state)) not in self.prior_state_dict.keys():
                    self.prior_state_dict[tuple(next_state)] = []
                    self.prior_state_dict[tuple(next_state)].append((tuple(start_state), action))
                else:
                    if (tuple(start_state), action) not in self.prior_state_dict[tuple(next_state)]:
                        self.prior_state_dict[tuple(next_state)].append((tuple(start_state), action))
                if (tuple(start_state), action, tuple(next_state)) not in self.time_steps.keys():
                    self.time_steps[(tuple(start_state), action, tuple(next_state))] = time_1
            next_state_list += next_states
        for next_state in next_state_list:
            self.generate_rec(depth - 1, next_state)

    def rec_sub_step(self, cur_state, action, rec_step=False, log_bool=False):
        ret_bool = False
        cur_action = action
        if cur_state == ["Terminal"]:
            ret_bool = True
        else:
            if action != None and 0 in cur_state[action].c_i.keys() and cur_state[action].c_i[0] == 1.0:
                ret_bool = True
        if ret_bool:
            if not rec_step:
                cur_action = None
            else:
                return [cur_state]
        if cur_action == None:
            none_state = self.next_state_scheduler(cur_state, action)
            action_dist = self.t_g.ret_action_dist(none_state, prev_state=cur_state)
            sub_next_states = []
            for task_action in action_dist.keys():
                next_sched_state = self.next_state_task(none_state, task_action)
                sub_next_states.append(next_sched_state)
                new_reward = 0
                new_prob = action_dist[task_action]
                if next_sched_state == ["Terminal"]:
                    new_reward -= 100000
                else:
                    for sub_job in next_sched_state:
                        if sub_job.d_i == 0 and not (0 in sub_job.c_i and sub_job.c_i[0] == 1.0):
                            new_reward -= 10
                self.total_reward[tuple(next_sched_state)] = new_reward + self.total_reward[tuple(cur_state)]
                if tuple(next_sched_state) in self.total_prob.keys():
                    self.total_prob[tuple(next_sched_state)] += new_prob * self.total_prob[tuple(cur_state)]
                else:
                    self.total_prob[tuple(next_sched_state)] = new_prob * self.total_prob[tuple(cur_state)]
                if tuple(next_sched_state) not in self.sub_prev_states.keys():
                    self.sub_prev_states[tuple(next_sched_state)] = []
                    self.sub_prev_states[tuple(next_sched_state)].append((tuple(cur_state), action))
                else:
                    if (tuple(cur_state), action) not in self.sub_prev_states[tuple(next_sched_state)]:
                        self.sub_prev_states[tuple(next_sched_state)].append((tuple(cur_state), action))
                self.sub_time_steps[tuple(next_sched_state)] = 1 + self.sub_time_steps[tuple(cur_state)]
            return sub_next_states
        state_list = []
        sub_next_states = []
        t_g_states = [self.next_state_scheduler(cur_state, action)]
        for t_g_state in t_g_states:
            task_actions = self.t_g.ret_possible_actions(t_g_state, prev_state=cur_state)
            action_dist = self.t_g.ret_action_dist(t_g_state, prev_state=cur_state)
            sub_next_states = []
            for task_action in action_dist.keys():
                next_sched_state = self.next_state_task(t_g_state, task_action)
                tuple_1 = self.in_reached_states(tuple(next_sched_state))
                if tuple_1[0] == True:
                    next_sched_state = list(tuple_1[1])
                else:
                    self.sub_step_states.append(tuple(next_sched_state))
                sub_next_states.append(next_sched_state)
                new_reward = 0
                new_prob = action_dist[task_action]
                if next_sched_state == ["Terminal"]:
                    new_reward -= 100000
                else:
                    for sub_job in next_sched_state:
                        if sub_job.d_i == 0 and not (0 in sub_job.c_i and sub_job.c_i[0] == 1.0):
                            new_reward -= 10
                self.total_reward[tuple(next_sched_state)] = new_reward + self.total_reward[tuple(cur_state)]
                if tuple(next_sched_state) in self.total_prob.keys():
                    if (tuple(cur_state), task_action) not in self.sub_prev_states[tuple(next_sched_state)]:
                        self.total_prob[tuple(next_sched_state)] += new_prob * self.total_prob[tuple(cur_state)]
                else:
                    self.total_prob[tuple(next_sched_state)] = new_prob * self.total_prob[tuple(cur_state)]
                if tuple(next_sched_state) not in self.sub_prev_states.keys():
                    self.sub_prev_states[tuple(next_sched_state)] = []
                    self.sub_prev_states[tuple(next_sched_state)].append((tuple(cur_state), task_action))
                else:
                    if (tuple(cur_state), task_action) not in self.sub_prev_states[tuple(next_sched_state)]:
                        self.sub_prev_states[tuple(next_sched_state)].append((tuple(cur_state), task_action))
                self.sub_time_steps[tuple(next_sched_state)] = 1 + self.sub_time_steps[tuple(cur_state)]
        for next_state in sub_next_states:
            state_list += self.rec_sub_step(next_state, action, rec_step=True, log_bool=log_bool)
        return state_list

    def in_reached_states(self, state):
        if self.sub_step_states == []:
            return (False, None)
        if state == tuple(["Terminal"]):
            if state in self.sub_step_states:
                return (True, tuple(["Terminal"]))
            else:
                return (False, None)
        for other_state in self.sub_step_states:
            if self.check_if_equal(state, other_state):
                return (True, other_state)
        return (False, None)

    def ret_start_state(self):
        output = []
        for task_1 in self.task_list:
            output.append(task_1.generate_job())
        out_1 = tuple(output)
        out_2 = self.check_in_states(output)
        return out_2[1]

    def possible_prior(self, state):
        prior_states = self.prior_state_dict[state]
        return prior_states

    def prune_state_actions(self):
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
        next_states = []
        if scheduler_bool:
            next_states.append(self.next_state_dict_s[(tuple(state), action_list[0])])
        else:
            next_states.append(self.next_state_dict_t[(tuple(state), action_list[0])])
        return next_states

    def possible_next_pruned(self, state, action):
        next_rewards = {}
        next_state_1 = tuple(self.next_state_dict_s[(tuple(state), action)])
        next_states_2 = []
        for action in self.state_actions[next_state_1]:
            next_state_3 = self.check_in_states_s(self.next_state_task(next_state_1, action))[1]
            next_states_2.append(next_state_3)
            next_rewards[next_states_2] = self.get_reward(next_state_1, action, next_state_3)
        return next_states_2, next_rewards

    def obtain_chain(self, starting_state, action_set):
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
        if state == tuple(["Terminal"]):
            if rew:
                return state, False, 1, self.get_reward(state, None, tuple(["Terminal"]))
            else:
                return state, False, 1
        total_reward = 0
        bool_1 = self.check_in_states_s(state)
        if not bool_1[0]:
            return
        state_1 = bool_1[1]
        possible_next = self.possible_next(state_1, action_list=[action])[0]
        prob_dict = {}
        for next_state_index in range(0, len(possible_next)):
            prob_dict[next_state_index] = self.transition(state_1, action, possible_next[next_state_index])
        next_index = np.random.choice(prob_dict.keys(), p=prob_dict.values())
        final_state = possible_next[next_index]
        time_val = self.time_steps[(state_1, action, final_state)]
        total_reward += self.get_reward(state_1, action, final_state)
        if final_state != tuple(["Terminal"]) and final_state != ["Terminal"]:
            if action !=  None:
                if final_state[action].finish():
                    out_bool = True
                else:
                    out_bool = False
            else:
                out_bool = False
        else:
            out_bool = False
        if rew == False:
            return final_state, out_bool, time_val
        else:
            return final_state, out_bool, time_val, total_reward

    def scheduler_step_pruned(self, state, action, prev_state = None, imp_index=-1):
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
        action_list = self.t_g.ret_possible_actions(state, prev_state=prev_state)
        if (all(action in self.state_actions[state] for action in action_list)):
            action_1 = tuple(self.t_g.ret_action(state))
        else:
            action_1 = self.state_actions[state][0]
        next_state = self.next_state_task(state, action_1)
        return next_state

    def task_gen_step(self, state, rew_1=False):
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

    def transition(self, state, action, next_state):
        prob_1 = self.check_in_states_s(state)
        prob_2 = self.check_in_states_s(next_state)
        if prob_1[0] and prob_2[0]:
            if (prob_1[1], action, prob_2[1]) in self.prob.keys():
                return self.prob[(prob_1[1], action, prob_2[1])]
            else:
                return 0
        else:
            return 0

    def get_reward(self, state, action, next_state):
        # Probability design decision
        prob_1 = self.check_in_states_s(state)
        prob_2 = self.check_in_states_s(next_state)
        if prob_1[0] and prob_2[0]:
            if (prob_1[1], action, prob_2[1]) in self.reward.keys():
                return self.reward[(prob_1[1], action, prob_2[1])]
            else:
                return 0
        else:
            return 0

    def value_iteration(self, conv_parameter):
        """Documentation for the value_iteration method:

            This method is a loop for the value calculation method. It takes a k parameter and iterates the
            value_calculation method k times.
        """
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
                    add_prob = self.transition(state, action, next_state)
                    add_reward = self.get_reward(state, action, next_state)
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

    def possible_next_scheduler(self, state, action):
        next_states = self.possible_next(state, action_list=[action])[0]
        out_dict = {}
        rew_dict = {}
        for next_state in next_states:
            next_3 = self.check_in_states_s(next_state)[1]
            rew_dict[next_3] = self.get_reward(state, action, next_3)
            if next_3 not in out_dict:
                out_dict[next_3] = self.transition(state, action, next_3)
            else:
                out_dict[next_3] += self.transition(state, action, next_3)
        return out_dict, rew_dict

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
                    action_values[index_1] += (self.transition(state, action, next_state) * (
                            self.get_reward(state, action, next_state) + (
                            self.discount * self.value[self.state_list_scheduler.index(next_state)])))
            policy_1[state] = self.state_actions[state][action_values.index(max(action_values))]
        return policy_1

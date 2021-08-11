import random
from task import task
import math
import numpy
from task_gen_MDP import task_gen_mdp

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


    def MCTS(self, state, depth, samples, discount = 0.5, rand=True):
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
        max_val = max(val_list)
        for key, value in self.act_val.items():
            if value == max_val:
                out_action = key
                break
        return out_action, max_val

    def MCTS_simulation_step(self, cur_state, action, depth, cur_reward, discount = 1.0, rand=True):
        if depth == 0:
            total_reward = cur_reward
            return total_reward
        next_state, total_reward = self.estimate_MDP.next_state_scheduler_mod(tuple(cur_state), action)
        next_state_1 = self.estimate_MDP.check_in_states_s(tuple(next_state))[1]
        if rand:
            next_action = random.choice(list(range(0, len(next_state_1))))
        else:
            if next_state_1 == tuple(["Terminal"]):
                next_action = None
            else:
                hard_deadline_dict = {}
                soft_deadline_dict = {}
                for job_index in range(0, len(cur_state)):
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
        new_task_list = []
        for task_index in self.prob_estimate_a:
            new_task_list.append(task(self.prob_estimate_c[task_index], self.MDP.task_list[task_index].d_i, self.prob_estimate_a[task_index], hard=self.MDP.task_list[task_index].hard))
        self.estimate_MDP = task_gen_mdp(new_task_list)
        self.estimate_MDP.generate_MDP(depth)
        return self.estimate_MDP

    def optimal_policy(self, conv_param):
        self.estimate_MDP.value_iteration(conv_param)
        out_pol = self.estimate_MDP.set_policy()
        return out_pol

    def test_optimal_policy(self, policy, num_ep=1):
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
        reward_dict = {}
        reward = 0
        for episode in range(1, 1 + num_ep):
            finished_bool = False
            temp_state = self.estimate_MDP.ret_start_state()
            start_state = self.estimate_MDP.check_in_states_s(temp_state)[1]
            current_state = self.estimate_MDP.check_in_states_s(temp_state)[1]
            while not finished_bool:
                action, out_val = self.MCTS(current_state, depth, num_samples, rand=rand)
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


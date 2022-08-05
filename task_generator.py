import numpy as np
import itertools

class task_generator:
    def __init__(self, task_list):
        """Documentation for the __init__ method:

                This method initializes a task_generator object. The task_generator replaces jobs when the inter-arrival
                time reaches 0. The task generator has 4 possible actions it can take for a given job. It can declare
                the job as finished ('fin'), replace a complete job with a new instance ('sub'), replace an incomplete
                job with a new instance ('killANDsub'), and do nothing ('e'). These actions are stored in the action list.
                The task generator takes a list of tasks in the system as an argument. The c_last parameter is used when
                choosing actions in ret_action, ret_possible_actions, and ret_action_dist.

        """
        self.actions = ["fin", "sub", "killANDsub", "e"]
        self.c_last = []
        self.task_list = task_list
        for task in task_list:
            self.c_last.append({})

    def ret_action(self, current_state):
        """Documentation for the ret_action method: This function returns a set of actions given a state. The
        current_state is defined as a list of jobs [j1, j2, ..., jn]. The function iterates over all the jobs in a
        state and returns an action for each job. If the completion time for a job has reached 0 a non-zero
        probability the function returns a 'fin' for the job based on the probability distribution. If the job has an
        inter-arrival time of 0, the function returns a 'sub' action depending on the inter-arrival distribution.
        Similarly, if the inter-arrival time is 0 with a non-zero probability and the job is incomplete, the function
        will return 'killANDsub' depending on the inter-arrival time distribution. Finally, if system is in the
        terminal state, or the computation time and inter-arrival times are not 0, the function will return a 'e'
        action for that job. The function returns a list of actions [a1, a2, ...., an] with each action in the list
        corresponding to a job in the current_state argument.


        """
        action_list = []
        action_dict = {}
        for job_index in range(0, len(current_state)):
            job = current_state[job_index]
            if job == "Terminal":
                for task in self.task_list:
                    action_list.append("e")
                break
            if 0 in job.c_i.keys():
                c_prob = job.c_i[0]
            else:
                c_prob = 0
            if 0 in job.a_i.keys():
                a_prob = job.a_i[0]
            else:
                a_prob = 0
            action_dict["fin"] = c_prob * (1 - a_prob)
            action_dict["sub"] = c_prob * a_prob
            action_dict["e"] = 0
            action_dict["killANDsub"] = (1 - c_prob) * a_prob
            if c_prob != 1:
                action_dict["e"] = (1 - c_prob) * (1 - a_prob)
            elif 0 in self.c_last[job_index].keys() and self.c_last[job_index][0] == 1:
                action_dict["e"] = (1 - a_prob)
                action_dict["fin"] = 0
            action_out = np.random.choice(action_dict.keys(), p=action_dict.values())
            if action_out == "killANDsub" or action_out == "sub":
                self.c_last[job_index] = {}
            action_list.append(action_out)
            self.c_last[job_index] = job.c_i
        return action_list

    def ret_possible_actions(self, current_state, prev_state = None):
        """Documentation for the ret_possible_actions method:

        This function is very similar to the ret_action function. However, unlike the ret_action function which only
        returns one list of actions. This function returns a list of all possible action combinations for the given
        state. For example if the current state has a job that will complete with probability p, this function will
        return a set of two possible action lists, one with a 'fin' action for the job and one with a 'e' action for
        the job. The prev_state argument is used to distinguish cases where a job finished in the last time step and
        prevents the function from returning 'fin' unnecessarily.

        """
        action_list = []
        for job_index in range(0, len(current_state)):
            action_dict = {}
            sub_list = []
            job = current_state[job_index]
            if job == "Terminal":
                for task in self.task_list:
                    action_list.append("e")
                break
            if 0 in job.c_i.keys():
                c_prob = job.c_i[0]
            else:
                c_prob = 0
            if 0 in job.a_i.keys():
                a_prob = job.a_i[0]
            else:
                a_prob = 0
            action_dict["fin"] = c_prob * (1 - a_prob)
            action_dict["sub"] = c_prob * a_prob
            action_dict["killANDsub"] = (1 - c_prob) * a_prob
            if prev_state == None:
                if c_prob != 1 and a_prob != 1:
                    action_dict["e"] = (1 - c_prob) * (1 - a_prob)
                elif 0 in self.c_last[job_index].keys() and self.c_last[job_index][0] == 1:
                    action_dict["e"] = (1 - a_prob)
                    action_dict["fin"] = 0
            else:
                if c_prob != 1 and a_prob != 1:
                    action_dict["e"] = (1 - c_prob) * (1 - a_prob)
                elif 0 in prev_state[job_index].c_i.keys() and prev_state[job_index].c_i[0] == 1:
                    action_dict["e"] = (1 - a_prob)
                    action_dict["fin"] = 0
            for key in action_dict.keys():
                if action_dict[key] > 0:
                    sub_list.append(key)
            action_list.append(sub_list)
            self.c_last[job_index] = job.c_i
        output = list(itertools.product(*action_list))
        return output

    def ret_action_dist(self, current_state, prev_state = None):
        """Documentation for the ret_action_dist method:

                This function returns a distribution of possible task_generator actions with associated probability.
                Just as with the ret_possible_actions function, this function returns all possible actions from a
                given state. In addition, the function also returns the probability of that action occurring. This
                probability is dependent on the computation time distribution and inter-arrival time distributions.
        """
        action_list = []
        action_dict_1 = {}
        action_dict = []
        for job_index in range(0, len(current_state)):
            action_dict.append({})
            sub_list = []
            job = current_state[job_index]
            if job == "Terminal":
                for task in self.task_list:
                    action_list.append("e")
                action_dict_1[tuple(action_list)] = 1.0
                return action_dict_1
            if 0 in job.c_i.keys():
                c_prob = job.c_i[0]
            else:
                c_prob = 0
            if 0 in job.a_i.keys():
                a_prob = job.a_i[0]
            else:
                a_prob = 0
            action_dict[job_index]["e"] = 0.0
            action_dict[job_index]["fin"] = c_prob * (1 - a_prob)
            action_dict[job_index]["sub"] = c_prob * a_prob
            action_dict[job_index]["killANDsub"] = (1 - c_prob) * a_prob
            if prev_state == None:
                if c_prob != 1:
                    action_dict[job_index]["e"] = (1 - c_prob) * (1 - a_prob)
                elif 0 in self.c_last[job_index].keys() and self.c_last[job_index][0] == 1:
                    action_dict[job_index]["e"] = (1 - a_prob)
                    action_dict[job_index]["fin"] = 0
            else:
                if c_prob != 1:
                    action_dict[job_index]["e"] = (1 - c_prob) * (1 - a_prob)
                elif 0 in prev_state[job_index].c_i.keys() and prev_state[job_index].c_i[0] == 1:
                    action_dict[job_index]["e"] = (1 - a_prob)
                    action_dict[job_index]["fin"] = 0
            for key in action_dict[job_index].keys():
                if action_dict[job_index][key] > 0:
                    sub_list.append(key)
            action_list.append(sub_list)
            self.c_last[job_index] = job.c_i
        output = list(itertools.product(*action_list))
        for action_index in range(0, len(output)):
            action_dict_1[tuple(output[action_index])] = 1
            for action_index_2 in range(0, len(output[action_index])):
                action = output[action_index][action_index_2]
                action_dict_1[tuple(output[action_index])] *= action_dict[action_index_2][action]
        return action_dict_1

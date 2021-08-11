from Gridworld import Gridworld
from MDP import MDP
from MarkovChain import MarkovChain
from agent import agent
from Clustering import Clustering
import numpy as np
import math
import json
import random
from data_MDP import data_MDP
import matplotlib.pyplot as plt
from task import task
from job import job
import itertools

class task_generator:
    def __init__(self, task_list):
        self.actions = ["fin", "sub", "killANDsub", "e"]
        self.c_last = []
        self.task_list = task_list
        for task in task_list:
            self.c_last.append({})
        self.a_last = {}

    def ret_action(self, current_state):
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
            # print("pre_case: ", c_prob, a_prob)
            if c_prob != 1:
                # print("case_1")
                action_dict["e"] = (1 - c_prob) * (1 - a_prob)
            elif 0 in self.c_last[job_index].keys() and self.c_last[job_index][0] == 1:
                # print("case_2")
                action_dict["e"] = (1 - a_prob)
                action_dict["fin"] = 0
            # print("action_dict: ", action_dict)
            action_out = np.random.choice(action_dict.keys(), p=action_dict.values())
            if action_out == "killANDsub" or action_out == "sub":
                self.c_last[job_index] = {}
            action_list.append(action_out)
            self.c_last[job_index] = job.c_i
        return action_list

    def ret_possible_actions(self, current_state, prev_state = None):
        action_list = []
        action_dict = {}
        for job_index in range(0, len(current_state)):
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
                if c_prob != 1:
                    action_dict["e"] = (1 - c_prob) * (1 - a_prob)
                elif 0 in self.c_last[job_index].keys() and self.c_last[job_index][0] == 1:
                    action_dict["e"] = (1 - a_prob)
                    action_dict["fin"] = 0
            else:
                if c_prob != 1:
                    action_dict["e"] = (1 - c_prob) * (1 - a_prob)
                elif 0 in prev_state[job_index].c_i.keys() and prev_state[job_index].c_i[0] == 1:
                    action_dict["e"] = (1 - a_prob)
                    action_dict["fin"] = 0
            # # print(c_prob, a_prob, action_dict)
            # action_out = np.random.choice(action_dict.keys(), p=action_dict.values())
            # if action_out == "killANDsub" or action_out == "sub":
            #     self.c_last[job_index] = {}
            # action_list.append(action_out)
            for key in action_dict.keys():
                if action_dict[key] > 0:
                    sub_list.append(key)
            action_list.append(sub_list)
            self.c_last[job_index] = job.c_i
        output = list(itertools.product(*action_list))
        return output

    def ret_action_dist(self, current_state, prev_state = None):
        action_list = []
        action_dict_1 = {}
        action_dict = []
        # print("current_state: ", current_state)
        for job_index in range(0, len(current_state)):
            action_dict.append({})
            sub_list = []
            job = current_state[job_index]
            # print("job: ", job)
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
            # print("output[action_1]: ", output[action_index])
            # print("action_dict: ", action_dict)
            for action_index_2 in range(0, len(output[action_index])):
                action = output[action_index][action_index_2]
                # print("action: ", action, action_dict[action_index_2][action])
                action_dict_1[tuple(output[action_index])] *= action_dict[action_index_2][action]
        # print("action_dict_1: ", action_dict_1)
        return action_dict_1
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

class job:
    def __init__(self, comp_time, deadline, inter_arrival, hard=True):
        self.c_i = comp_time
        self.d_i = deadline
        self.a_i = inter_arrival
        self.h_s = hard
        self.finished = False
        self.failed = False
        self.past_deadline = False
        self.terminal = True


    def step(self, action_val):
        if action_val == True:
            new_c_i = {}
            for key in self.c_i.keys():
                if key > 0:
                    new_c_i[key - 1] = self.c_i[key]
                else:
                    new_c_i[key] = self.c_i[key]
            self.c_i = new_c_i
        if self.d_i > 0:
            self.d_i -= 1
        new_a_i = {}
        for key_a in self.a_i.keys():
            if key_a > 0:
                new_a_i[key_a - 1] = self.a_i[key_a]
            else:
                new_a_i[key_a] = self.a_i[key_a]
        if new_a_i == {}:
            new_a_i[0] = 1
        self.a_i = new_a_i

    def step_full(self, num_time_steps, action_val):
        if action_val == True:
            new_c_i = {0 :1.0}
            new_d_i = self.d_i - num_time_steps
            new_a_i = {}
            for key in self.a_i.keys():
                key_a = key - num_time_steps
                new_a_i[key_a] = self.a_i[key]
        else:
            new_c_i = self.c_i
            new_d_i = self.d_i - num_time_steps
            new_a_i = {}
            for key in self.a_i.keys():
                key_a = key - num_time_steps
                new_a_i[key_a] = self.a_i[key]
        self.c_i = new_c_i
        self.d_i = new_d_i
        self.a_i = new_a_i


    def finish(self):
        if 0 in self.c_i.keys() and self.c_i[0] == 1:
            return True
        else:
            return False

    def return_data(self):
        if self.failed:
            return "Terminal"
        return tuple([self.c_i, self.d_i, self.a_i])

    def is_same(self, other):
        # # print("current: ", self.c_i, self.d_i, self.a_i)
        # # print("other: ", other.c_i, other.d_i, other.a_i)
        if self.c_i == other.c_i and self.d_i == other.d_i and self.a_i == other.a_i: #and self.failed == other.failed and self.past_deadline == other.past_deadline:
            # # print("returning True: ", )
            # # print(" ")
            return True
        # # print("returning False")
        # # print(" ")
        return False

    def return_key(self):
        return (tuple(sorted(self.c_i.items())), self.d_i, tuple(sorted(self.a_i.items())))
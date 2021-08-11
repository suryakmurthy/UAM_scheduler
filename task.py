from Gridworld import Gridworld
from MDP import MDP
from MarkovChain import MarkovChain
from agent import agent
from Clustering import Clustering
import numpy as np
import math
import json
from data_MDP import data_MDP
import matplotlib.pyplot as plt
from job import job
from prob_dist import prob_dist

class task:
    def __init__(self, comp_time, deadline, inter_arrival, hard=True):
        self.c_i = comp_time
        self.d_i = deadline
        self.a_i = inter_arrival
        self.hard = hard

    def generate_job(self):
        # print("generating job: ", self.hard)
        output = job(self.c_i, self.d_i, self.a_i, hard=self.hard)
        return output
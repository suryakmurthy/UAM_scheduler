
class job:
    def __init__(self, comp_time, deadline, inter_arrival, hard=True):
        """Documentation for the __init__ method:
                This method initializes the job object. The function takes a comp_time distribution, deadline, and interarrival time its arguments.
                The job class has variable associated with computation time (c_i), deadline (d_i) and inter-arrival time (a_i).
                The function also has a "hard" argument which dictates if the job is hard or soft. The function also initializes several global variables.
                The "finished" variable is True if the job is completed, the "failed" variable is True if the job is hard and misses its deadline, and
                the "past_deadline" variable to check if the job is past its deadline.
        """
        self.c_i = comp_time
        self.d_i = deadline
        self.a_i = inter_arrival
        self.h_s = hard
        self.finished = False
        self.failed = False
        self.past_deadline = False


    def step(self, action_val):
        """Documentation for the step method:
                This method steps the job forward by one time step. The action_val argument is true if the job is being worked on during this time step.
                If action value is true, the function decrements the computation time, deadline, and inter-arrival time parameters. If action value is
                false, it only decrements the deadline and inter-arrival time.

        """
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
        """Documentation for the step_full method:
                This method steps the job forward by a set number of time steps supplied by the argument num_time_steps. The action_val argument is true if the job is being worked on during this time step.
                If action value is true, the function decrements the computation time, deadline, and inter-arrival time parameters. If action value is
                false, it only decrements the deadline and inter-arrival time.

        """
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
        """Documentation for the finish method:
                This function returns True if the computation time is 0 with a 1.0 probability. Otherwise, the function returns False.

        """
        if 0 in self.c_i.keys() and self.c_i[0] == 1:
            return True
        else:
            return False

    def return_data(self):
        """Documentation for the return_data method:
                This function returns the computation time, deadline, and inter-arrival time of the job.
        """
        if self.failed:
            return "Terminal"
        return tuple([self.c_i, self.d_i, self.a_i])

    def is_same(self, other):
        """Documentation for the is_same method:
                This function compares the job object with another job object given in the argument "other".
                If the two objects are equal (Same computation time, deadline, and inter-arrival time), the
                function returns True. Otherwise, it returns False.
        """
        if self.c_i == other.c_i and self.d_i == other.d_i and self.a_i == other.a_i:
            return True
        return False

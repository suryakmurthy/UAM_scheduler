from job import job


class task:
    def __init__(self, comp_time, deadline, inter_arrival, hard=True):
        """Documentation for the __init__ method:
                This method initializes a task object. Just like job objects, tasks have a computation time (c_i),
                a deadline (d_i), and an inter-arrival time (a_i). Tasks are also denoted as hard or soft, determined by
                the "hard" parameter.
        """
        self.c_i = comp_time
        self.d_i = deadline
        self.a_i = inter_arrival
        self.hard = hard

    def generate_job(self):
        """Documentation for the generate_job method:
                The generate_job method generates a job object using the existing parameters. The function
                returns the new job object as "output"
        """
        output = job(self.c_i, self.d_i, self.a_i, hard=self.hard)
        return output

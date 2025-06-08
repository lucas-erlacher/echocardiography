class BetaScheduler():
    # this code enables the use of all beta schedules from the following paper: https://arxiv.org/abs/1903.10145
    # (both the presented schedule and the ones that they are comparing themselves to)
    # 
    # mode has to be one of the following: ["const", "cyclic"]
    # in case mode="const" the value needs to be provided under "const_value"
    # in case mode="cyclic" 
    # - the cycle period needs to be provided under "cyclic_period" 
    # - the maximum value of beta under "cyclic_max" 
    # - the minimum value of beta under "cyclic_max" 
    # - the number of steps that the increase should take under cyclic_num_increasing
    def __init__(self, mode, const_val=None, cyclic_period=None, cyclic_num_increasing=None, cyclic_min=None, cyclic_max=None):
        self.mode = mode
        self.const_val = const_val
        self.cyclic_period = cyclic_period
        self.cyclic_num_increasing = cyclic_num_increasing  
        self.cyclic_min = cyclic_min
        self.cyclic_max = cyclic_max
        # compute additional quantities that will be relevant for the beta computation in cyclic mode
        if self.mode == "cyclic":
            self.increase_range = self.cyclic_max - self.cyclic_min  # magnitude of how much beta will increase over cyclic_min
            self.per_step_increase = self.increase_range / self.cyclic_num_increasing  # how much we will increase beta in an additional timestep (in increase phase)

    def get_beta(self, timestep):
        if self.mode == "const":
            return self.const_val
        elif self.mode == "cyclic":
            curr_period_timestep = timestep % self.cyclic_period  # magnitude of timestep that indexes into current period (ignoring everything that indexed into earlier periods)
            if curr_period_timestep < self.cyclic_num_increasing:  # we are in increase phase
                # since we are in curr_period_timestep we need curr_period_timestep many increases of magnitude self.per_step_increase
                curr_increase = self.per_step_increase * curr_period_timestep  
                return self.cyclic_min + curr_increase
            else:  # we are in constant phase
                return self.cyclic_max
            
    def get_summary(self):
        s = "mode=" + str(self.mode) + ", "
        if self.mode == "const":
            s += "const_val=" + str(self.const_val)
        elif self.mode == "cyclic":
            s += "cyclic_period=" + str(self.cyclic_period) + ", "
            s += "cyclic_num_increasing=" + str(self.cyclic_num_increasing) + ", "
            s += "cyclic_min=" + str(self.cyclic_min) + ", "
            s += "cyclic_max=" + str(self.cyclic_max)
        return s
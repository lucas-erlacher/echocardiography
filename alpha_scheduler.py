# simple alpha scheduler (currently only supports constant and linearly decreasing alpha schedules)

class AlphaScheduler:
    def __init__(self, max, min, timesteps):
        self.max = max
        self.min = min
        self.timesteps = timesteps

    def get_alpha(self, epoch_num):
        if epoch_num > self.timesteps: return self.min  # the schedule is done
        else: return self.min + (self.max - self.min) * epoch_num  # we are in the schedule
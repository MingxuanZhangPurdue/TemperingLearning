import torch
import math


class Scheduler:
    def __init__(self, T, init_sigma):
        """
        T: int
        init_sigma: float
        """
        assert T > 0, "T must be greater than 0"
        assert init_sigma > 0, "init_sigma must be greater than 0"
        
        self.T = T
        self.init_sigma = init_sigma
        self.sigmas = self._calculate_sigmas()
    
    def _calculate_sigmas(self):
        raise NotImplementedError("Subclasses must implement _calculate_sigmas method")
    
    def add_noise(self, t, y):
        assert 0 <= t < self.T, "t must be less than T"
        if t == self.T - 1:
            return y
        else:
            return y + torch.randn_like(y) * self.sigmas[t]
    
    def transform(self, y):
        return torch.stack([self.add_noise(t, y) for t in range(self.T)])
    

class LinearScheduler(Scheduler):
    def _calculate_sigmas(self):
        return [self.init_sigma - (self.init_sigma * t) / (self.T-1) for t in range(self.T)]
    
class PowerScheduler(Scheduler):
    def __init__(self, T, init_sigma, power):
        self.power = power
        super().__init__(T, init_sigma)
    
    def _calculate_sigmas(self):
        return [self.init_sigma * ((1 - t / (self.T - 1)) ** self.power) for t in range(self.T)]
    

class StepwiseScheduler(Scheduler):
    def __init__(self, T, init_sigma, num_steps):
        self.num_steps = num_steps
        super().__init__(T, init_sigma)
    
    def _calculate_sigmas(self):
        step_size = self.T // self.num_steps
        sigmas = []
        for step in range(self.num_steps):
            sigma = self.init_sigma * (1 - step / self.num_steps)
            sigmas.extend([sigma] * step_size)
        return sigmas[:self.T] 
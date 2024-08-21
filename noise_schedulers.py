import torch


class LinearScheduler:

    def __init__(self, T, init_sigma):

        '''
        T: int
        init_sigma: float
        '''

        assert T > 0, "T must be greater than 0"
        assert init_sigma > 0, "start_sigma must be greater than 0"
        
        self.T = T
        self.init_sigma = init_sigma
        self.sigmas = [self.init_sigma - (self.init_sigma * t) / (self.T-1) for t in range(self.T)]
        
    def add_noise(self, t, y):
        assert 0 <= t < self.T, "t must be less than or equal to T"
        # don't add noise when t == T-1
        if t == self.T - 1:
            return y
        # add noise when t < T
        else:
            return y + torch.randn_like(y) * self.sigmas[t]
        
    def transform(self, y):

        D = torch.stack([self.add_noise(t, y) for t in range(self.T)])

        return D
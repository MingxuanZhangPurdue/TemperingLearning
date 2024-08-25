import torch
import numpy as np
from tqdm import tqdm

class LinearLRScheduler:
    def __init__(self, num_iters, init_factor=1.0, end_factor=0.0):
        self.init_factor = init_factor
        self.end_factor = end_factor
        self.num_iters = num_iters
        self.decrement = (init_factor - end_factor) / num_iters

    def get_factor(self, current_iter):
        if current_iter >= self.num_iters:
            return self.end_factor
        return self.init_factor - self.decrement * current_iter

class TemperingLearningRegression:

    def __init__(
        self,
        D, X,
        model,
        MC_steps,
        sigmas, zeta,
        init_lr, init_factor=1.0, end_factor=0.1,
        n=0.1, m=0.5,
        burn_in_fraction = 0.2,
        tau = 1.0,
        X_test=None, y_test=None):

        # make sure model, X, and D are on the same device
        assert X.device == D.device and model.parameters().__next__().device == D.device, "X, model, and D must be on the same device"
        if X_test is not None and y_test is not None:
            assert X.device == X_test.device and X_test.device == y_test.device, "train and test sets must be on the same device"

        # torch tensor with shape (T, N, 1)
        self.D = D
        # torch tensor with shape (N, p)
        self.X = X
        # torch model
        self.model = model
        # a list of state dictionaries
        self.S_prev = []
        # a list of state dictionaries
        self.S_next = []
        # torch tensor of shape (T)
        self.sigmas = sigmas
        # set tau
        self.tau = tau
        # set zeta
        self.zeta = zeta
        # set burn_in period
        assert 0 < burn_in_fraction < 1, "burn_in_fraction must be greater than 0 and less than 1"
        self.burn_in_fraction = burn_in_fraction
        self.burn_in_steps = int(burn_in_fraction * MC_steps)
        # get T and N
        self.T, self.N, _ = D.shape
        # set MC_steps
        self.MC_steps = MC_steps
        # calculate the number of MC samples
        self.n_MC = MC_steps - self.burn_in_steps
        # set n, if n is fraction then convert it to integer, else check if it is greater than 0 and less than or equal to N
        self.n = int(n * self.N) if 0 < n < 1 and type(n) else n
        assert 0 < self.n <= self.N, "n must be greater than 0 and less or eqaul to N"
        # set m
        self.m = int(m * self.n_MC) if 0 < m < 1 else m
        assert 0 < self.m <= self.n_MC, "m must be greater than 0 and less or eqaul to self.n_MC"
        # set test set
        self.X_test = X_test
        self.y_test = y_test
        # set up lr scheduler
        self.init_lr = init_lr
        self.lr_scheduler = LinearLRScheduler(self.T, init_factor=init_factor, end_factor=end_factor)
    
    def subsampling(self, t):

        '''
        t: int, the time index
        '''

        assert 0 <= t < self.T, "t must be greater than or equal to 0 and less than T"

        y_ids = torch.randperm(self.N)[:self.n]

        if t > 0:
            assert len(self.S_prev) == self.n_MC, "the length of self.S_prev must be equal to self.n_MC"
            theta_ids = torch.randperm(self.n_MC)[:self.m].tolist()
        else:
            theta_ids = None

        return {"t": t, "y_ids": y_ids, "theta_ids": theta_ids}

    def gradient_estimation(self, ids_dict):
        '''
        ids_dict: a dictionary with three keys: t, y_ids, and theta_ids
        this function updates the gradients of the self.model
        '''
        # make sure the gradients of the model are zero
        self.model.zero_grad()

        t = ids_dict["t"]
        y_ids = ids_dict["y_ids"]
        theta_ids = ids_dict["theta_ids"]

        assert 0 <= t < self.T, "t must be greater than or equal to 0 and less than T"
        assert t == 0 or len(self.S_prev) == self.n_MC, "the length of self.S_prev must be equal to self.n_MC"

        # x: torch tensor with shape (n, p)
        x = self.X[y_ids, :]
        # y: torch tensor with shape (n, 1)
        y = self.D[t, y_ids, :]
        # y_hat: torch tensor with shape (n, 1)
        y_hat = self.model(x)

        mse_loss = self.N * torch.nn.functional.mse_loss(y_hat, y, reduction="mean") / (2 * (self.sigmas[t]**2 + self.tau**2))
        mse_loss.backward()

        # add additional gradients for t > 0
        if t > 0:
            with torch.no_grad():
                # Stack the list of tensors with arbitrary shape (_, _, ....) into a single tensor with shape (m, _, _, ....)
                sampled_thetas = [self.S_prev[i] for i in theta_ids]
                for n, p in self.model.named_parameters():
                    thetas_p = torch.stack([theta[n] for theta in sampled_thetas])
                    # Calculate the difference between the current parameter and the stacked parameters
                    diff = p - thetas_p
                    # Flatten the difference tensor along all dimensions except the first (shape: (m, -1))
                    diff_flattened = diff.view(diff.shape[0], -1) #diff.flatten(start_dim=1)
                    # Compute the squared norm of the flattened differences and apply the exponential function (shape: (m))
                    exp_norm_squared = torch.exp(-torch.sum(diff_flattened**2, dim=1) / (2 * self.zeta**2))
                    # Reshape to ensure correct broadcasting for element-wise multiplication (shape: (m, 1, 1, ....))
                    exp_norm_squared = exp_norm_squared.view(-1, *([1] * (diff.dim() - 1)))
                    # Compute the weighted sum of differences (shape: (_, _, ....))
                    weighted_diff = torch.sum(exp_norm_squared * diff, dim=0)
                    # Compute the sum of the exponential weights (shape: (1))
                    denominator = exp_norm_squared.sum()
                    # Update the gradient of the parameter
                    p.grad -= (1 / self.zeta**2) * weighted_diff / denominator
    
    def parameter_update(self, lr):
        with torch.no_grad():
            for p in self.model.parameters():
                p.sub_(lr * p.grad).add_(torch.randn_like(p) * np.sqrt(2 * lr))

    def sample_collection(self):
        self.S_next.append({k: v.clone() for k, v in self.model.state_dict().items()})
    
    def evaluation(self, X, y):
        self.model.eval()
        with torch.no_grad():
            y_hat = self.model(X)
            mse = torch.nn.functional.mse_loss(y_hat, y, reduction="mean").item()
        self.model.train()
        return mse

    def train(self):

        train_loss = []
        test_loss = []

        for t in tqdm(range(self.T)):

            self.S_prev = self.S_next
            self.S_next = []

            lr = self.lr_scheduler.get_factor(t) * self.init_lr

            for l in range(self.MC_steps):

                # subsampling
                ids_dict = self.subsampling(t)
                # gradient estimation
                self.gradient_estimation(ids_dict)
                # parameter update
                self.parameter_update(lr)
                # sample collection
                if l >= self.burn_in_steps:
                    self.sample_collection()
            
            # evaluation
            train_mse_loss = self.evaluation(self.X, self.D[-1,:,:])
            train_loss.append(train_mse_loss)
            #tqdm.write(f"Train MSE Loss at iteration {t}: {train_mse_loss}")

            if self.X_test is not None and self.y_test is not None:
                test_mse_loss = self.evaluation(self.X_test, self.y_test)
                test_loss.append(test_mse_loss)
                #tqdm.write(f"Test MSE Loss at iteration {t}: {test_mse_loss}")
        return train_loss, test_loss

            
            
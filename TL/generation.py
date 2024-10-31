import torch
import math
import lightning as L
from typing import Union

class Tempering(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        noise_scheduler,
        num_training_samples: int,
        T: int,
        num_mc_steps: int,      
        zeta: float = 1.0,
        mc_subset_ratio: Union[float, int] = 0.5,
        burn_in_fraction: float = 0.7,
        random_walk_step_size: float = 0.0,
    ):
        super().__init__()
        
        self.model = model

        self.noise_scheduler = noise_scheduler

        assert 1 <= T <= self.noise_scheduler.config.num_train_timesteps, "T must be greater or equal to 1 and less or equal to the number of training timesteps of the noise scheduler"
        self.T = T
        self.timestep_schedule = torch.linspace(0, self.noise_scheduler.config.num_train_timesteps - 1, T, dtype=torch.long)

        # calculate the cumulative product of reversed alphas
        self.alphas_reversed_cumprod = torch.cumprod(torch.flip(self.noise_scheduler.alphas, [0]), dim=0)
        # each prod starts from the current timestep to the end
        self.alphas_reversed_cumprod = torch.flip(self.alphas_reversed_cumprod, [0])

        # set zeta
        assert zeta > 0, "zeta must be greater than 0"
        self.zeta = zeta

        # set the number of Monte Carlo steps
        assert num_mc_steps >= 1, "num_mc_steps must be greater or equal to 1"
        self.num_mc_steps = num_mc_steps    

        # set the number of burn-in steps
        assert 0 <= burn_in_fraction < 1, "burn_in_fraction must be greater or equal to 0 and less than 1"
        self.burn_in_fraction = burn_in_fraction
        self.burn_in_steps = int(burn_in_fraction * self.num_mc_steps)

        # calculate the number of Monte Carlo samples
        self.num_mc_samples = self.num_mc_steps - self.burn_in_steps
        assert self.num_mc_samples > 0, "n_MC must be greater than 0, please adjust burn_in_fraction or num_mc_steps accordingly!"

        # set the size of the Monte Carlo subset that we randomly select from the Monte Carlo sample buffers
        self.mc_subset_size = int(mc_subset_ratio * self.num_mc_samples) if 0 < mc_subset_ratio < 1 and type(mc_subset_ratio) == float else mc_subset_ratio
        assert 0 < self.mc_subset_size <= self.num_mc_samples, "mc_subset_size must be greater than 0 and less or equal to self.num_mc_samples, please adjust mc_subset_ratio accordingly!"

        # set therandom walk step size
        assert random_walk_step_size >= 0, "random_walk_step_size must be greater or equal to 0"
        self.random_walk_step_size = random_walk_step_size

        # set the number of training samples
        self.num_training_samples = num_training_samples

        # a list of state dictionaries
        self.S_prev = []
        # a list of state dictionaries
        self.S_next = []
    

    def add_noise_from_timestep(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:

        self.alphas_reversed_cumprod = self.alphas_reversed_cumprod.to(device=original_samples.device)
        alphas_reversed_cumprod = self.alphas_reversed_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_reversed_cumprod = alphas_reversed_cumprod[timesteps] ** 0.5
        sqrt_alpha_reversed_cumprod = sqrt_alpha_reversed_cumprod.flatten()
        while len(sqrt_alpha_reversed_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_reversed_cumprod = sqrt_alpha_reversed_cumprod.unsqueeze(-1)

        sqrt_one_minus_alpha_reversed_cumprod = (1 - alphas_reversed_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_reversed_cumprod = sqrt_one_minus_alpha_reversed_cumprod.flatten()
        while len(sqrt_one_minus_alpha_reversed_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_reversed_cumprod = sqrt_one_minus_alpha_reversed_cumprod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_reversed_cumprod * original_samples + sqrt_one_minus_alpha_reversed_cumprod * noise
        return noisy_samples
    

    def training_step_epsilon_prediction(self, batch, t):

        assert 0 <= t <= self.T-1, "t must be in the range of 0 to self.T-1, including the boundary values!"

        # get the clean images
        clean_images = batch["images"]

        # add noise to the clean images to generate noisy images
        if t == self.T-1:
            noisy_images = clean_images  
        else:
            timestep = self.timestep_schedule[self.T - t - 2].to(clean_images.device)
            noisy_images = self.noise_scheduler.add_noise(clean_images, torch.randn_like(clean_images), timestep)

        # draw epsilon from the standard normal distribution
        epsilon = torch.randn_like(clean_images)

        # add epsilon to the noisy images to generate pesudo white (standard normal) noise from there
        if t == self.T-1:
            timestep = self.timestep_schedule[0].to(clean_images.device)
        else:
            timestep = self.timestep_schedule[self.T - t - 2].to(clean_images.device)
        pesudo_white_noise = self.add_noise_from_timestep(noisy_images, epsilon, timestep)

        # predict the epsilon
        predicted_epsilon = self.model(pesudo_white_noise, t).sample

        # calculate the mean squared error loss
        mse_loss = torch.nn.functional.mse_loss(epsilon, predicted_epsilon)/2
        
        return mse_loss

    def training_step(self, batch, t):
        return self.training_step_epsilon_prediction(batch, t)
    

    def on_before_optimizer_step(self, t):
        # add additional gradients for t > 0
        if t > 0:
            # check if the Monte Carlo sample buffers are collected enough
            assert len(self.S_prev) == self.num_mc_samples, "the length of self.S_prev must be equal to self.num_mc_samples, make sure to collect enough samples!"
            # randomly select m samples from the Monte Carlo sample buffers
            theta_ids = torch.randperm(self.num_mc_samples, device=self.device)[:self.mc_subset_size].tolist()
            with torch.no_grad():
                # get the selected samples
                sampled_thetas = [self.S_prev[i] for i in theta_ids]
                # calculate the additional gradients for the model parameters
                for n, p in self.model.named_parameters():
                    # Stack the list of tensors with arbitrary shape (_, _, ....) into a single tensor with shape (m, _, _, ....)
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
                    weighted_diff = torch.einsum('i...,i...->...', exp_norm_squared, diff)
                    # Compute the sum of the exponential weights (shape: (1))
                    denominator = exp_norm_squared.sum() + 1e-8
                    # Update the gradient of the parameter
                    p.grad -= (1 / (self.zeta**2)) * weighted_diff / (denominator * self.num_training_samples)


    def on_train_batch_end(self, lr, batch_idx):
        # add random walk step for the model parameters
        if self.random_walk_step_size > 0:
            with torch.no_grad():
                for p in self.model.parameters():
                    p.add_(torch.randn_like(p) * math.sqrt(2 * lr * self.random_walk_step_size))
        
        # collect the model parameters as Monte Carlo samples
        if batch_idx >= self.burn_in_steps:
            self.S_next.append({k: v.detach().clone() for k, v in self.model.state_dict().items()})


    def reset_sample_buffers(self):
        # reset the Monte Carlo sample buffers
        self.S_prev = self.S_next
        self.S_next = []


    def generate_samples(self, batch_size, t=None):

        assert t is None or 0 <= t <= self.T - 1, "t must be in the range of 0 to self.T-1"

        t = self.T-1 if t is None else t

        # Generate initial noise
        white_noise = torch.randn(
            (batch_size, self.model.config.in_channels, self.model.config.sample_size, self.model.config.sample_size),
            device=self.model.device
        )

        # get the timestep
        if t == 0:
            timestep = self.timestep_schedule[-2].to(white_noise.device)
        else:
            timestep = self.timestep_schedule[self.T - t - 1].to(white_noise.device)

        # Predict epsilon for the first timestep
        predicted_epsilon = self.model(white_noise, t).sample

        # Get the cumulative product of alphas for the last timestep
        alpha_reversed_prod = self.alphas_reversed_cumprod[timestep].to(device=white_noise.device)

        # Calculate beta_prod
        beta_reversed_prod = 1 - alpha_reversed_prod

        # Predict the original sample
        pred_original_sample = (white_noise - beta_reversed_prod**0.5 * predicted_epsilon) / alpha_reversed_prod**0.5

        return pred_original_sample

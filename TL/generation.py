import torch
import math
import lightning as L


class Tempering(L.LightningModule):
    def __init__(
        self,
        model,
        noise_scheduler,
        MC_steps,
        num_training_samples,
        zeta=1.0,
        m=0.5,
        burn_in_fraction=0.5,
        random_walk_step_size=0.0,
        prediction_type="epsilon",
    ):
        super().__init__()

        self.model = model

        self.noise_scheduler = noise_scheduler
        self.alphas = self.noise_scheduler.alphas
        self.alphas_reversed_cumprod = torch.cumprod(torch.flip(self.alphas, [0]), dim=0)
        # each prod starts from the current timestep to the end
        self.alphas_reversed_cumprod = torch.flip(self.alphas_reversed_cumprod, [0])

        self.T = noise_scheduler.config.num_train_timesteps

        assert zeta > 0, "zeta must be greater than 0"
        self.zeta = zeta

        # set MC_steps
        self.MC_steps = MC_steps

        # set burn in fraction
        assert 0 <= burn_in_fraction < 1, "burn_in_fraction must be greater or equal to 0 and less than 1"
        self.burn_in_fraction = burn_in_fraction
        self.burn_in_steps = int(burn_in_fraction * self.MC_steps)

        # calculate the number of MC samples
        self.n_MC = self.MC_steps - self.burn_in_steps
        assert self.n_MC > 0, "n_MC must be greater than 0, please adjust burn_in_fraction or MC_steps accordingly!"

        # set m
        self.m = int(m * self.n_MC) if 0 < m < 1 and type(m) == float else m
        assert 0 < self.m <= self.n_MC, "m must be greater than 0 and less or eqaul to self.n_MC"

        # random walk step size
        assert random_walk_step_size >= 0, "random_walk_step_size must be greater or equal to 0"
        self.random_walk_step_size = random_walk_step_size


        self.prediction_type = prediction_type

        self.num_training_samples = num_training_samples


        self.S_prev = []
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

        sqrt_alpha_prod = alphas_reversed_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_reversed_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    

    def training_step_epsilon_prediction(self, batch, t):

        clean_images = batch["images"]

        if t == self.T-1:
            noisy_images = clean_images  
        elif t < self.T-1:
            noisy_images = self.noise_scheduler.add_noise(clean_images, torch.randn_like(clean_images), torch.tensor([self.T - t - 2], device=clean_images.device, dtype=torch.int64))
        else:
            raise ValueError(f"Invalid timestep: {t}, please choose from 0 to {self.T-1}")

        epsilon = torch.rand_like(clean_images)

        pure_noise = self.add_noise_from_timestep(noisy_images, epsilon, torch.tensor([self.T - t - 1], device=clean_images.device, dtype=torch.int64))

        predicted_epsilon = self.model(pure_noise, t).sample

        mse_loss = torch.nn.functional.mse_loss(epsilon, predicted_epsilon)
        
        return self.num_training_samples * mse_loss
    
    def training_step_sample_prediction(self, batch, t):

        clean_images = batch["images"]

        if t >= self.T-1:
            noisy_images = clean_images  
        else:
            noisy_images = self.noise_scheduler.add_noise(clean_images, torch.randn_like(clean_images), torch.tensor([self.T - t - 2], device=clean_images.device, dtype=torch.int64))

        epsilon = torch.rand_like(clean_images)

        pure_noise = self.add_noise_from_timestep(noisy_images, epsilon, torch.tensor([self.T - t - 1], device=clean_images.device, dtype=torch.int64))

        predicted_noisy_images = self.model(pure_noise, t).sample

        mse_loss = torch.nn.functional.mse_loss(noisy_images, predicted_noisy_images)    

        return self.num_training_samples * mse_loss


    def training_step(self, batch, t):

        if self.prediction_type == "epsilon":
            return self.training_step_epsilon_prediction(batch, t)
        elif self.prediction_type == "sample":
            return self.training_step_sample_prediction(batch, t)
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}, please choose from ['epsilon', 'sample']")
    

    def on_before_optimizer_step(self, t):

        # add additional gradients for t > 0
        if t > 0:

            assert len(self.S_prev) == self.n_MC, "the length of self.S_prev must be equal to self.n_MC, make sure to collect enough samples!"
            theta_ids = torch.randperm(self.n_MC, device=self.device)[:self.m].tolist()
        
            with torch.no_grad():
                sampled_thetas = [self.S_prev[i] for i in theta_ids]
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
                    #weighted_diff = torch.sum(exp_norm_squared * diff, dim=0)
                    # Compute the sum of the exponential weights (shape: (1))
                    denominator = exp_norm_squared.sum() + 1e-8
                    # Update the gradient of the parameter
                    p.grad -= (1 / (self.zeta**2)) * weighted_diff / denominator


    def on_train_batch_end(self, lr, batch_idx):

        if self.random_walk_step_size > 0:
            with torch.no_grad():
                for p in self.model.parameters():
                    p.add_(torch.randn_like(p) * math.sqrt(2 * lr * self.random_walk_step_size))
        
        if batch_idx >= self.burn_in_steps:
            self.S_next.append({k: v.detach().clone() for k, v in self.model.state_dict().items()})


    def reset_sample_buffers(self):
        self.S_prev = self.S_next
        self.S_next = []


    def generate_samples(self, batch_size):

        # Generate initial noise
        noise = torch.randn(batch_size, *self.model.image_size, device=self.model.device)

        # Predict epsilon for the first timestep
        predicted_epsilon = self.model(noise, torch.tensor([self.T-1], device=self.model.device, dtype=torch.int64)).sample

        # Get the cumulative product of alphas for the last timestep
        alpha_prod = self.alphas_cumprod[-1]

        # Calculate beta_prod
        beta_prod = 1 - alpha_prod

        # Predict the original sample
        pred_original_sample = (noise - beta_prod**0.5 * predicted_epsilon) / alpha_prod**0.5

        return pred_original_sample

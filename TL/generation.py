import torch
import math
import lightning as L


class TemperingGeneration(L.LightningModule):
    def __init__(
        self, 
        model,
        optimizer,
        lr_scheduler,
        noise_scheduler,
        num_training_batches,
        lr_scheduler_interval="step",
        zeta=1.0,
        m=0.2,
        burn_in_fraction=0.5,
        random_walk_step_size=0,
        prediction_type="noise",
    ):
        super().__init__()

        self.model = model

        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        assert lr_scheduler_interval in ["step", "epoch"], "lr_scheduler_interval must be either 'step' or 'epoch'"
        self.lr_scheduler_interval = lr_scheduler_interval

        self.noise_scheduler = noise_scheduler
        self.alphas = self.noise_scheduler.alphas
        self.alphas_reversed_cumprod = torch.cumprod(torch.flip(self.alphas, [0]), dim=0)
        # each prod starts from the current timestep to the end
        self.alphas_cumprod = torch.flip(self.alphas_reversed_cumprod, [0])

        self.T = noise_scheduler.config.num_train_timesteps

        assert zeta > 0, "zeta must be greater than 0"
        self.zeta = zeta

        # set MC_steps
        self.MC_steps = num_training_batches

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


        self.S_prev = []
        self.S_next = []

    
    def subsampling(self, t):

        if t > 0:
            assert len(self.S_prev) == self.n_MC, "the length of self.S_prev must be equal to self.n_MC"
            theta_ids = torch.randperm(self.n_MC, device=self.device)[:self.m].tolist()
        else:
            theta_ids = []

        return {"t": t, "theta_ids": theta_ids}
    

    def add_noise_from_timestep(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    

    def training_step_noise_prediction(self, batch, batch_idx):

        t = self.current_epoch

        clean_images = batch["images"]

        if t >= self.T-1:
            noisy_images = clean_images  
        else:
            noisy_images = self.noise_scheduler.add_noise(clean_images, torch.randn_like(clean_images), torch.tensor([self.T - t - 2], device=clean_images.device, dtype=torch.int64))

        noise = torch.rand_like(clean_images)

        pure_noise_images = self.add_noise_from_timestep(noisy_images, noise, torch.tensor([self.T - t - 1], device=clean_images.device, dtype=torch.int64))

        predicted_noise = self.model(pure_noise_images, t).sample

        loss = torch.nn.functional.mse_loss(noise, predicted_noise)

        self.log("train_loss", loss, prog_bar=True)
        
        return loss
    
    def training_step_sample_prediction(self, batch, batch_idx):

        t = self.current_epoch

        clean_images = batch["images"]

        if t >= self.T-1:
            noisy_images = clean_images  
        else:
            noisy_images = self.noise_scheduler.add_noise(clean_images, torch.randn_like(clean_images), torch.tensor([self.T - t - 2], device=clean_images.device, dtype=torch.int64))

        noise = torch.rand_like(clean_images)

        pure_noise_images = self.add_noise_from_timestep(noisy_images, noise, torch.tensor([self.T - t - 1], device=clean_images.device, dtype=torch.int64))

        predicted_samples = self.model(pure_noise_images, t).sample

        loss = torch.nn.functional.mse_loss(pure_noise_images, predicted_samples)

        self.log("train_loss", loss, prog_bar=True)
        
        return loss


    def training_step(self, batch, batch_idx):

        if self.prediction_type == "noise":
            return self.training_step_noise_prediction(batch, batch_idx)
        elif self.prediction_type == "sample":
            return self.training_step_sample_prediction(batch, batch_idx)
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")
    

    def on_before_optimizer_step(self, optimizer):

        t = self.current_epoch
        ids_dict = self.subsampling(t)
        theta_ids = ids_dict["theta_ids"]

        # add additional gradients for t > 0
        if t > 0:
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


    def on_train_batch_end(self, outputs, batch, batch_idx):

        if self.random_walk_step_size > 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            with torch.no_grad():
                for p in self.model.parameters():
                    p.add_(torch.randn_like(p) * math.sqrt(2 * lr * self.random_walk_step_size))
        
        if batch_idx >= self.burn_in_steps:
            self.sample_collection()


    def on_train_epoch_start(self):
        self.S_prev = self.S_next
        self.S_next = []


    def sample_collection(self):
        self.S_next.append({k: v.detach().clone() for k, v in self.model.state_dict().items()})


    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.lr_scheduler
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": self.lr_scheduler_interval, "frequency": 1}}

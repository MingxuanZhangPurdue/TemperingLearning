import torch
import torchvision
import argparse

from dataclasses import dataclass
from lightning.fabric.utilities import AttributeDict
from wandb.integration.lightning.fabric import WandbLogger
from tqdm import tqdm

import lightning as L

from PIL import Image
from diffusers import DDPMScheduler
from datasets import load_dataset
from torchvision import transforms
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from TL.generation import Tempering

def parse_args():

    parser = argparse.ArgumentParser(description="Tempering Learning for Unconditional Image Generation")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes to use")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "ddp", "fsdp", "dp"])
    parser.add_argument("--precision", type=str, default=None, choices=["32", "32-true", "16-true", "16-mixed", "bf16-true", "bf16-mixed", None])

    parser.add_argument("--image_size", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)


    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--num_epochs_per_timestep", type=int, default=3, help="number of epochs per timestep")
    parser.add_argument("--burn_in_fraction", type=float, default=0.9)
    parser.add_argument("--zeta", type=float, default=4.0)
    parser.add_argument("--m", type=float, default=0.5)
    parser.add_argument("--random_walk_step_size", type=float, default=0.0)
    parser.add_argument("--sample_prediction", action="store_true", help="whether to use sample prediction")

    parser.add_argument("--wandb_project_name", type=str, default="Tempering Learning Unconditional Image Generation")

    args = parser.parse_args()

    return args

def save_images(images: torch.Tensor, filename: str):
    images = (images / 2 + 0.5).clamp(0, 1).squeeze()
    grid = torchvision.utils.make_grid(images, nrow=5)
    grid_np = (grid.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    grid_pil = Image.fromarray(grid_np)
    grid_pil.save(filename)

def fit(args):

    # set seed
    L.seed_everything(args.seed)

    # set wandb logger
    wandb_logger = WandbLogger(
        project=args.wandb_project_name
    )
    wandb_logger.log_hyperparams(vars(args))

    # set fabric
    fabric = L.Fabric(
        accelerator="auto", 
        devices="auto",
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision=args.precision,
        loggers=[wandb_logger]
    )

    # launch fabric
    fabric.launch()

    # set dataset
    train_dataset = load_dataset(
        "huggan/smithsonian_butterflies_subset", 
        split="train", 
        cache_dir="./cache"
    )

    # set transform
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    train_dataset.set_transform(transform)
    num_training_samples = len(train_dataset)

    # set dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True,
        pin_memory=True,
    )
    num_training_batches = len(train_dataloader)

    print ("Number of training batches:", num_training_batches)

    # set model
    model = UNet2DModel(
        sample_size=args.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(32, 32, 64, 64, 128, 128),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    num_trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_trainable_params}")
    
    # set noise scheduler
    noise_scheduler = DDPMScheduler()
    print ("Number of training timesteps:", noise_scheduler.config.num_train_timesteps)

    # set tempering learning
    # set tempering learning
    num_mc_steps = num_training_batches * args.num_epochs_per_timestep
    tlmodel = Tempering(
        model=model,
        noise_scheduler=noise_scheduler,
        T=args.T,
        num_mc_steps=num_mc_steps,
        num_training_samples=num_training_samples,
        zeta=args.zeta,
        mc_subset_ratio=args.m,
        burn_in_fraction=args.burn_in_fraction,
        random_walk_step_size=args.random_walk_step_size,
        sample_prediction=args.sample_prediction,
    )

    print ("Number of MC steps:", tlmodel.num_mc_steps)
    print ("Burn in steps:", tlmodel.burn_in_steps)
    print ("Number of MC samples:", tlmodel.num_mc_samples)
    print ("MC subset size:", tlmodel.mc_subset_size)

    print ("Number of training samples:", num_training_samples)
    print ("Zeta:", args.zeta)
    print ("Random walk step size:", args.random_walk_step_size)

    print ("timestep mapping:", tlmodel.timestep_schedule)

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate
    )

    # set lr scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.T*args.num_epochs_per_timestep*num_training_batches
    )

    # set up objects
    tlmodel, optimizer = fabric.setup(tlmodel, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # train
    tlmodel.train()

    progress_bar = tqdm(range(args.T), desc="Training timesteps")

    for t in progress_bar:

        tlmodel.reset_sample_buffers()

        batch_idx = 0

        avg_train_loss_per_timestep = 0

        for epoch in range(args.num_epochs_per_timestep):

            for _, batch in enumerate(train_dataloader):

                optimizer.zero_grad()

                loss = tlmodel.training_step(batch, t)

                avg_train_loss_per_timestep += loss.item()

                fabric.backward(loss)

                tlmodel.on_before_optimizer_step(t)

                optimizer.step()

                lr_scheduler.step()

                tlmodel.on_train_batch_end(lr_scheduler.get_last_lr()[0], batch_idx)

                batch_idx += 1

        avg_train_loss_per_timestep /= (args.num_epochs_per_timestep * num_training_batches)
        progress_bar.set_postfix({'Avg train loss': f'{avg_train_loss_per_timestep:.4f}'})
        fabric.log_dict({
            "avg_train_loss_per_timestep": avg_train_loss_per_timestep,
        })

        tlmodel.eval()
        images = tlmodel.generate_samples(args.train_batch_size, t)
        save_images(images, f"generated_samples/images_{t}.png")
        tlmodel.train()

    state = AttributeDict(model=model)
    fabric.save("output/model.pt", state)

if __name__ == "__main__":
    args = parse_args()
    fit(args)

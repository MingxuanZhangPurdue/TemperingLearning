import torch
import argparse
import lightning as L

from diffusers import DDPMScheduler
from datasets import load_dataset
from torchvision import transforms
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

from TL.generation import TemperingGeneration



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes to use")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "ddp", "fsdp", "dp"])
    parser.add_argument("--precision", type=str, default=None, choices=["32", "32-true", "16-true", "16-mixed", "bf16-true", "bf16-mixed", None])

    parser.add_argument("--image_size", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)


    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--num_epochs_per_timestep", type=int, default=2, help="number of epochs per timestep")
    parser.add_argument("--burn_in_fraction", type=float, default=0.5)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=0.5)
    parser.add_argument("--prediction_type", type=str, default="epsilon")
    parser.add_argument("--random_walk_step_size", type=float, default=0)

    args = parser.parse_args()

    return args

def fit(args):

    # set seed
    L.seed_everything(args.seed)

    # set fabric
    fabric = L.Fabric(
        accelerator="auto", 
        devices="auto",
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision=args.precision,
    )

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

    # set noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.T)

    # set tempering learning
    MC_steps = num_training_batches * args.num_epochs_per_timestep
    tlmodel = TemperingGeneration(
        model=model,
        noise_scheduler=noise_scheduler,
        MC_steps=MC_steps,
        num_training_samples=num_training_samples,
        zeta=args.zeta,
        m=args.m,
        burn_in_fraction=args.burn_in_fraction,
        random_walk_step_size=args.random_walk_step_size,
        prediction_type=args.prediction_type,
    )

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

    for t in tqdm(range(args.T), desc="Training timesteps"):

        tlmodel.reset_sample_buffers()

        batch_idx = 0

        avg_train_loss_per_timestep = 0

        for epoch in range(args.num_epochs_per_timestep):

            for _, batch in enumerate(train_dataloader):

                optimizer.zero_grad()

                loss = tlmodel.training_step(batch, t)

                avg_train_loss_per_timestep += loss.item()/num_training_samples

                fabric.backward(loss)

                tlmodel.on_before_optimizer_step(t)

                optimizer.step()

                lr_scheduler.step()

                tlmodel.on_train_batch_end(lr_scheduler.get_last_lr()[0], batch_idx)

                batch_idx += 1

        avg_train_loss_per_timestep /= (args.num_epochs_per_timestep * num_training_batches)
        tqdm.set_postfix({'Avg train loss': f'{avg_train_loss_per_timestep:.4f}'})

if __name__ == "__main__":
    args = parse_args()
    fit(args)

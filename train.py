import argparse
import lightning as L
from TL.generation import TemperingGeneration



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_size", type=int, default=32)


    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)

def fit():

    fabric = L.Fabric(...)

    # Instantiate the LightningModule
    model = TemperingGeneration()

    # Get the optimizer(s) from the LightningModule
    optimizer = model.configure_optimizers()

    # Get the training data loader from the LightningModule
    train_dataloader = model.train_dataloader()

    # Set up objects
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # Call the hooks at the right time
    model.on_train_start()

    model.train()
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model.training_step(batch, i)
            fabric.backward(loss)
            optimizer.step()

            # Control when hooks are called
            if condition:
                model.any_hook_you_like()
import torch
import numpy as np
import random
import wandb
import argparse

from noise_schedulers import LinearScheduler
from tempering import TemperingLearningRegression
from models import MLP
from preprocess import preprocess_UCI_dataset

name_to_id_dict = {
    "wine_quality": 186,
    "student_performance": 320,
    "abalone": 1,
    "liver_disorders": 60,
    "concrete_compressive_strength": 165
}

def parse_arguments():

    parser = argparse.ArgumentParser(description='Hyperparameter tuning for tempering learning on UCI regression tasks')

    parser.add_argument(
        '--method', 
        type=str, 
        default='grid',
        choices=['random', 'grid'],
        help='Method for hyperparameter tuning'
    )

    parser.add_argument(
        '--count', 
        type=int, 
        default=None,
        help='Number of hyperparameter combinations to try'
    )

    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='wine_quality',
        choices=name_to_id_dict.keys(),
        help='Name of the dataset to preprocess'
    )

    parser.add_argument(
        "--encoding_type",
        type=str,
        default="onehot",
        choices=["onehot", "ordinal"],
        help="Type of encoding to use for categorical features"
    )

    parser.add_argument(
        "--normalize_target",
        action="store_true",
        help="Whether to normalize the target variable"
    )

    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.2,
        help='Proportion of the dataset to include in the test split'
    )

    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--hidden_dim', 
        type=int, 
        default=32,
        help='Hidden dimension of the MLP'
    )

    parser.add_argument(
        '--num_layers', 
        type=int, 
        default=2,
        help='Number of layers of the MLP'
    )

    return parser.parse_args()


def fit(config=None):

    with wandb.init(config=config):

        config = wandb.config

        noise_scheduler = LinearScheduler(T=config.T, init_sigma=config.init_sigma)
        D = noise_scheduler.transform(y_train)
        sigmas = noise_scheduler.sigmas

        model = MLP(X_train.shape[-1], [config.hidden_dim] * config.num_layers, 1).to(device)

        trainer = TemperingLearningRegression(
            D = D,
            X = X_train,
            model = model,
            sigmas = sigmas,
            tau = config.tau,
            zeta = config.zeta,
            init_lr = config.lr,
            init_factor = 1.0,
            end_factor = config.end_factor,
            burn_in_fraction=config.burn_in_fraction,
            MC_steps=config.MC_steps,
            n = config.n,
            m = config.m,
            X_test = X_test,
            y_test = y_test,
            logger = wandb,
            progress_bar = False
        )

        trainer.train()



if __name__ == "__main__":


    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    wandb.login()

    print ("Specified dataset name:", args.dataset_name)
    print ("UCI ID:", name_to_id_dict[args.dataset_name])
    print ("Test size:", args.test_size)
    print ("Encoding type:", args.encoding_type)
    print ("Device: ", device)
    print ("Seed: ", args.seed)

    X_train, X_test, y_train, y_test, preprocessor, y_scaler = preprocess_UCI_dataset(
        name_to_id_dict[args.dataset_name], 
        args.encoding_type, 
        args.normalize_target,
        args.test_size, 
        args.seed
    )
    
    print ("X_train: ", X_train.shape)
    print ("X_test: ", X_test.shape)
    print ("y_train: ", y_train.shape)
    print ("y_test: ", y_test.shape)

    # convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


    sweep_config = {
        'method': args.method if hasattr(args, 'method') else 'random'
    }

    metric = {
        'name': 'loss',
        'goal': 'minimize'   
        }

    sweep_config['metric'] = metric

    parameters_dict = {
        'T': {
            'values': [10, 20, 50, 100, 200]
            },
        'init_sigma': {
            'values': [2.0, 1.0, 0.5, 0.1]
        },
        'lr': {
            'values': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        },
        'MC_steps': {
            'values': [10, 20, 50, 100, 200]
        },
        'burn_in_fraction': {
            'value': 0.5
        },
        'zeta': {
            'value': 1.0
        },
        'end_factor': {
            'value': 0.1
        },
        'n': {
            'value': 0.1
        },
        'm': {
            'value': 0.5
        },
        'tau': {
            'value': 1.0
        },
        'hidden_dim': {
            'value': args.hidden_dim
        },
        'num_layers': {
            'value': args.num_layers
        }
    }

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=f"TL-{args.dataset_name}")

    wandb.agent(sweep_id, fit, count=args.count)





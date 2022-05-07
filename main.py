from cgi import test
from model import MLPModel
import sys
import torch  # noqa
import wandb
from pathlib import Path
from torch import nn  # noqa
from torch.utils.data import DataLoader
from dataset import OnevsMNIST
from train import Trainer
from utils import get_hyperparameters


if __name__ == '__main__':
    debug = True
    hyperparams = get_hyperparameters(sys.argv)
    for optim in hyperparams["optimizers"]:
        print("Running optimizer: " + str(optim))

        for regularization in hyperparams["loss_func"]:
            print("Chosen loss function constrain: " + str(regularization))

            for cl in hyperparams["classes"]:
                print("Number of classes: " + str(cl))
                wandb.config = {  # noqa
                    "learning_rate": optim["params"]["lr"],
                    "weight_decay": optim["params"]["weight_decay"],
                    "epochs": optim["epochs"],
                    "batch_size": 64,
                    "embed_dim": 16,
                    "num_classes": cl,
                    "seeds": [0, 42, 1337, 3213, 9999, 5783, 1023],
                    "optimizer": optim["name"]

                }
                torch.use_deterministic_algorithms(True)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                criterion = nn.CrossEntropyLoss()
                optimizer = getattr(torch.optim, wandb.config['optimizer'])
                optimizer_params = optim["params"]

                log_dir = Path('./logs/' + 
                               wandb.config['optimizer'] + "/" + 
                               str(regularization["func"].__name__) + "/lambda=" 
                               + str(regularization["lambda"]))

                trainer = Trainer(
                    seeds=wandb.config['seeds'],
                    model=MLPModel,
                    optimizer=optimizer,
                    optimizer_params=optimizer_params,
                    criterion=criterion,
                    device=device,
                    dataset=OnevsMNIST,
                    num_classes=wandb.config['num_classes'],
                    regularization=regularization,
                    log_dir=log_dir
                    )
                wandbmode = 'disabled' if debug else 'online'
                trainer.train(
                    wandb=wandb,
                    wandbmode=wandbmode,
                    embed_dim=wandb.config['embed_dim'],
                    epochs=wandb.config['epochs'],
                    batch_size=wandb.config['batch_size'])

                del trainer
                del optimizer


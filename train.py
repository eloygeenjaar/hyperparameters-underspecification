import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from aggregator import Aggregator
from utils import *
from visualization import *


class Trainer:
    def __init__(self, seeds, model, optimizer, optimizer_params,
                 criterion, device, dataset, num_classes,
                 regularization, log_dir):
        self.seeds = seeds
        self.model = model
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.criterion = criterion
        self.device = device
        self.dataset = dataset
        self.num_classes = num_classes
        self.checkpoints = [None for i in range(len(seeds))]
        self.previous_best = np.inf
        self.softmax = nn.Softmax(dim=-1)
        self.regularization = regularization
        self.log_dir = log_dir
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.aggregator = Aggregator(seeds)
        self.current_seed = None

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_step(self, wandb, model, optimizer, x, y, current_step,epoch):
        optimizer.zero_grad()
        x = x.to(self.device)
        y = y.to(self.device)
        logit_pred, representation_tuple = model(x)
        loss = self.criterion(logit_pred, y)
        loss += self.regularization['func'](model.lin1.weight,
                                            self.regularization['lambda'])
        log = {'train-loss': loss.detach()}
        for i in range(self.num_classes):
            subset = (y == i)
            if logit_pred[subset].size(0) != 0:
                acc = (self.softmax(logit_pred[subset]).argmax(1)
                    == y[subset]).sum() / subset.size(0)
                log[f'train-{i}-acc'] = acc
        log['train-total-acc'] = (self.softmax(logit_pred).argmax(1)
                                  == y).sum() / x.size(0)
        loss.backward()
        self.aggregator(
            (self.current_seed, model, True),
            (self.current_seed, model, True),
            (self.current_seed, representation_tuple, 'train', epoch, y))
        optimizer.step()
        wandb.log(log)
        return loss.detach(), log['train-total-acc']

    def valid_step(self, wandb, model, optimizer, x, y, current_step, epoch):
        x = x.to(self.device)
        y = y.to(self.device)
        logit_pred, representation_tuple = model(x)
        loss = self.criterion(logit_pred, y)
        loss += self.regularization['func'](model.lin1.weight,
                                            self.regularization['lambda'])
        log = {'valid-loss': loss.detach()}
        for i in range(self.num_classes):
            subset = (y == i)
            if logit_pred[subset].size(0) != 0:
                acc = (self.softmax(logit_pred[subset]).argmax(1)
                    == y[subset]).sum() / subset.size(0)
                log[f'valid-{i}-acc'] = acc
        log['valid-total-acc'] = (self.softmax(logit_pred).argmax(1)
                                  == y).sum() / x.size(0)
        wandb.log(log)
        self.aggregator(
            (self.current_seed, model, False), 
            (self.current_seed, model, False),
            (self.current_seed, representation_tuple, 'val', epoch, y))
        return loss, log['valid-total-acc']

    def epoch_step(self, wandb, model, epoch, optimizer,
                   loaders, loader_type: str, seed_ix):
        loader = loaders[loader_type]
        # get function
        step = getattr(self, f'{loader_type}_step')
        current_step = 0
        loss = 0.0
        acc = 0.0
        for (i, (x, y)) in enumerate(tqdm(loader)):
            loss_step, acc_step = step(wandb, model, optimizer, x, y, current_step, epoch)
            loss += (loss_step * x.size(0))
            acc += (acc_step * x.size(0))
            current_step += 1
        print(f'{loader_type.capitalize()} Epoch: {epoch}, '
              f'loss: {loss / len(loader.dataset)},'
              f'acc: {acc / len(loader.dataset)}')
        if loader_type == 'valid' and loss <= self.previous_best:
            self.checkpoints[seed_ix] = model.state_dict()
            self.previous_best = loss

    def train(self, wandb, wandbmode, embed_dim, epochs, batch_size):
        seed_dirs = []
        for (seed_ix, seed) in enumerate(self.seeds):
            self.current_seed = seed
            seed_dir = self.log_dir / Path(
                f'model-{seed}-ed{embed_dim}-lr{self.optimizer_params["lr"]}-'
                f'nc{self.num_classes}')
            create_dir_safe(seed_dir)
            seed_dirs.append(seed_dir)
            wandb.init(project="stochasticity", entity="underspecification-gt",
                       reinit=True, name=f'hyperparams-{seed}-{self.num_classes}', mode=wandbmode)
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            model = self.model(
                embed_dim=embed_dim,
                num_classes=self.num_classes).to(self.device)
            optimizer = self.optimizer(model.parameters(),
                                       **self.optimizer_params)
            print(f'Training with seed: {seed}')
            wandb.watch(model, log='all')
            g = torch.Generator()
            g.manual_seed(seed)

            train_loader = DataLoader(
                self.dataset(
                    '../MNIST', num_classes=self.num_classes,
                    train=True, download=True),
                shuffle=True,
                batch_size=batch_size,
                pin_memory=True,
                worker_init_fn=self.seed_worker,
                generator=g)
            valid_loader = DataLoader(
                self.dataset(
                    '../MNIST', num_classes=self.num_classes,
                    train=False, download=True),
                shuffle=False,
                batch_size=batch_size,
                pin_memory=True,
                worker_init_fn=self.seed_worker,
                generator=g)
            print('Number of model parameters: '
                  f'{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

            loaders = {'train': train_loader,
                       'valid': valid_loader}

            for epoch in range(epochs):
                model.train()
                self.epoch_step(
                    wandb, model, epoch, optimizer,
                    loaders, 'train', seed_ix)
                model.eval()
                with torch.no_grad():
                    self.epoch_step(
                        wandb, model, epoch, optimizer,
                        loaders, 'valid', seed_ix)
            # Dump all parameters
            self.aggregator.dump(seed, seed_dir)
            del train_loader
            del valid_loader
            del loaders
            del g
            del model
            del optimizer
        viz_dir = self.log_dir / Path(
                f'model-viz-ed{embed_dim}-lr{self.optimizer_params["lr"]}-'
                f'nc{self.num_classes}')
        create_dir_safe(viz_dir)
        seed_dirs = []
        for (i, seed) in enumerate(self.seeds):
            seed_dir = self.log_dir / Path(
                f'model-{seed}-ed{embed_dim}-lr{self.optimizer_params["lr"]}-'
                f'nc{self.num_classes}')
            if not seed_dir.is_dir():
                seed_dir.mkdir(parents=True, exist_ok=True)
            seed_file = seed_dir / Path('model.pt')
            torch.save(self.checkpoints[i], seed_file)
            seed_dirs.append(seed_dir)
        calculate_permutation_correlation(
            seed_dirs, self.seeds, embed_dim, self.num_classes, viz_dir)

        visualizer = Visualizer(viz_dir, seed_dirs, self.seeds,
                                epochs, embed_dim,
                                self.num_classes)

        param_params = {"num_classes": self.num_classes}
        visualizer((None, ),
                   param_params,
                   (self.seeds, ))


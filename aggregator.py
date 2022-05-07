import torch
from torch import nn
from pathlib import Path
from utils import create_dir_safe
import numpy as np
import copy


class Aggregator:
    def __init__(self, seeds):
        self.seeds = seeds
        self.gradients = {seed: [] for seed in seeds}
        self.parameters = {seed: [] for seed in seeds}
        self.representations = {'train': {seed: [] for seed in seeds}, 'val': {seed: [] for seed in seeds}}
        self.activations = {'train': {seed: [] for seed in seeds}, 'val': {seed: [] for seed in seeds}}
        self.ys = {'train': {seed: [] for seed in seeds}, 'val': {seed: [] for seed in seeds}}
        self.epoch = 0
        self.repbuffer = []
        self.actsbuffer = []
        self.ybuffer = []

    def __call__(self, gradient_params, param_params, rep_params):
        self.aggregate_gradients(gradient_params)
        self.aggregate_parameters(param_params)
        self.aggregate_representations(rep_params)

    def aggregate_gradients(self, input_params):
        seed, model, needgradient = input_params
        # step_dict = {}
        # for (key, val) in model.named_parameters():
        #     step_dict[key] = val.grad
        if (needgradient):
            grad_dictionary = {key: val.grad for key,val in model.named_parameters()}
            self.gradients[seed].append(copy.deepcopy(grad_dictionary))

    def dump_gradients(self, seed, seed_dir):
        # Dumping gradients
        step_dict = {
            key: [] for key in self.gradients[seed][0].keys()
        }
        for model_dict in self.gradients[seed]:
            for key in model_dict.keys():
                step_dict[key].append(model_dict[key])
        for key in step_dict.keys():
            step_dict[key] = torch.stack(step_dict[key], dim=0)
        grad_file = seed_dir / 'gradients.pt'
        torch.save(step_dict, grad_file)
        print(f'Dumped {len(self.gradients[seed])} steps of parameters'
              f' for seed: {seed} in {grad_file}')
        self.gradients[seed].clear()

    def aggregate_parameters(self, input_params):
        seed, model, save = input_params
        if save:
            self.parameters[seed].append(copy.deepcopy(model).state_dict())

    def dump_parameters(self, seed, seed_dir):
        # Dumping parameters
        step_dict = {
            key: [] for key in self.parameters[seed][0].keys()
        }
        for model_dict in self.parameters[seed]:
            for key in model_dict.keys():
                step_dict[key].append(model_dict[key])
        for key in step_dict.keys():
            step_dict[key] = torch.stack(step_dict[key], dim=0)
        parameter_file = seed_dir / 'parameters.pt'
        torch.save(step_dict, parameter_file)
        print(f'Dumped {len(self.parameters[seed])} steps of parameters'
              f' for seed: {seed} in {parameter_file}')
        self.parameters[seed].clear()

    def aggregate_representations(self, input_params):
        # TODO: Add mores
        seed, rep_tuple, mode, epoch, y = input_params
        embeddings, activation = rep_tuple
        if epoch != self.epoch and len(self.repbuffer)>0:
            reps = np.concatenate(self.repbuffer)
            acts = np.concatenate(self.actsbuffer)
            ys = np.concatenate(self.ybuffer)
            self.representations[mode][seed].append(reps)  # change to equality if we only care about last representations
            self.activations[mode][seed].append(acts) 
            self.ys[mode][seed].append(ys)
            self.repbuffer = []
            self.actuffer = []  
            self.ybuffer = []
            self.epoch = epoch
        self.repbuffer.append(embeddings.detach().numpy())
        self.actsbuffer.append(activation.detach().numpy()) 
        self.ybuffer.append(y.detach().numpy())    

    def dump_representations(self, seed, seed_dir):
        t='train'
        v='val'
        for e in [t,v]:
            representations_file = seed_dir / (f'{e}representations.npy')
            activations_file = seed_dir / (f'{e}activations.npy')
            ys_file = seed_dir / (f'{e}y.npy')
            np.save(representations_file, self.representations[e][seed])
            np.save(activations_file, self.activations[e][seed])
            np.save(ys_file, self.ys[e][seed])

        print(f'Dumped {len(self.representations[t][seed])}  train steps and {len(self.representations[v][seed])} val steps of representations'
              f' for seed: {seed} in {representations_file}')

        print(f'Dumped {len(self.activations[t][seed])}  train steps and {len(self.activations[v][seed])} val steps of activations'
              f' for seed: {seed} in {activations_file}')
        self.representations[t][seed].clear()
        self.representations[v][seed].clear()
        self.activations[t][seed].clear()
        self.activations[v][seed].clear()


    def dump(self, seed, seed_dir):
        agg_dir = seed_dir / Path('aggregations')
        create_dir_safe(agg_dir)
        self.dump_gradients(seed, agg_dir)
        self.dump_parameters(seed, agg_dir)
        self.dump_representations(seed, agg_dir)


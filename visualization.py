import gif
import torch
import numpy as np
import matplotlib
from sklearn.decomposition import PCA
from utils import create_dir_safe
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from utils import *


class Visualizer:
    def __init__(self, logdir, seed_dirs, seeds,
                 num_epochs, embed_dim, num_classes):
        self.seed_dirs = seed_dirs
        self.seeds = seeds
        self.logdir = logdir # folder to save figures in
        self.num_epochs = num_epochs
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def __call__(self, gradient_params, param_params, rep_params):
        self.visualize_gradients(gradient_params)
        self.visualize_parameters(param_params)
        self.visualize_parameters_fig(param_params)
        self.visualize_representations(rep_params)

    def visualize_gradients(self, gradient_params):
        for i, seed_dir in enumerate(self.seed_dirs):
            agg_dir = seed_dir / Path('aggregations')
            step_dict = torch.load(agg_dir / 'gradients.pt')
            if i == 0:
                gradientsweights_dict = {
                    key: [step_dict[key].view(
                        step_dict[key].size(0), *return_shape_params(
                            key, self.embed_dim, self.num_classes)
                        )] for key in step_dict.keys()
                }
            else:
                for key in step_dict.keys():
                    weights = step_dict[key]
                    weights = weights.view(
                        weights.size(0), *return_shape_params(
                            key, self.embed_dim, self.num_classes))
                    gradientsweights_dict[key].append(weights)
        for key in gradientsweights_dict.keys():
            gradientsweights_dict[key] = torch.stack(gradientsweights_dict[key], dim=1)

        rows, columns = calc_rows_cols(self.embed_dim)
        for (key, weights) in gradientsweights_dict.items():
            num_steps = weights.size(0)
            steps_per_epoch = num_steps // self.num_epochs

            for (i, seed) in enumerate(self.seeds):
                vmin = weights[:, i].min()
                vmax = weights[:, i].max()
                frames = []
                # Create a frame every 4 times an epoch
                for j in range(0, num_steps, steps_per_epoch // 16):
                    frame = plot_gif_parameters(
                        rows, columns, self.embed_dim,
                        weights[j, i], vmin, vmax, key)
                    frames.append(frame)
                weights_dir = self.logdir / Path('gradient_weights')
                create_dir_safe(weights_dir)
                gif.save(frames,
                        weights_dir / Path(f'gradweights_{key}_{seed}.gif'),
                        duration=100)

            var_wts = weights.var(1)
            var_frames = []
            # Create a frame every 4 times an epoch
            for i in range(0, num_steps, steps_per_epoch // 16):
                var_frame = plot_gif_parameters(
                    rows, columns, self.embed_dim,
                    var_wts[i], vmin, vmax, key)
                var_frames.append(var_frame)
            gif.save(frames,
                     weights_dir / Path(f'var_{key}.gif'),
                     duration=100)
        print(f'Created gradient (weights) trajectory gifs')

    def visualize_parameters(self, param_params):
        for i, seed_dir in enumerate(self.seed_dirs):
            agg_dir = seed_dir / Path('aggregations')
            step_dict = torch.load(agg_dir / 'parameters.pt')
            if i == 0:
                parameter_dict = {
                    key: [step_dict[key].view(
                        step_dict[key].size(0), *return_shape_params(
                            key, self.embed_dim, self.num_classes)
                        )] for key in step_dict.keys()
                }
            else:
                for key in step_dict.keys():
                    params = step_dict[key]
                    params = params.view(
                        params.size(0), *return_shape_params(
                            key, self.embed_dim, self.num_classes))
                    parameter_dict[key].append(params)

        for key in parameter_dict.keys():
            parameter_dict[key] = torch.stack(parameter_dict[key], dim=1)

        rows, columns = calc_rows_cols(self.embed_dim)
        for (key, params) in parameter_dict.items():
            num_steps = params.size(0)
            steps_per_epoch = num_steps // self.num_epochs

            for (i, seed) in enumerate(self.seeds):
                vmin = params[:, i].min()
                vmax = params[:, i].max()
                frames = []
                # Create a frame every 4 times an epoch
                for j in range(0, num_steps, steps_per_epoch // 4):
                    frame = plot_gif_parameters(
                        rows, columns, self.embed_dim,
                        params[j, i], vmin, vmax, key)
                    frames.append(frame)
                parameter_dir = self.logdir / Path('parameters')
                create_dir_safe(parameter_dir)
                gif.save(frames,
                        parameter_dir / Path(f'params_{key}_{seed}.gif'),
                        duration=100)

            var_params = params.var(1)
            var_frames = []
            # Create a frame every 4 times an epoch
            for i in range(0, num_steps, steps_per_epoch // 4):
                var_frame = plot_gif_parameters(
                    rows, columns, self.embed_dim,
                    var_params[i], vmin, vmax, key)
                var_frames.append(var_frame)
            gif.save(frames,
                     parameter_dir / Path(f'var_{key}.gif'),
                     duration=100)
        print(f'Created parameter trajectory gifs')


    def visualize_representations(self, rep_params):
        seeds,  = rep_params
        num_seeds = len(self.seed_dirs)
        representations = {'train': {seed: [] for seed in self.seed_dirs}, 'val': {seed: [] for seed in self.seed_dirs}}
        activations = {'train': {seed: [] for seed in self.seed_dirs}, 'val': {seed: [] for seed in self.seed_dirs}}
        ys = {'train': {seed: [] for seed in self.seed_dirs}, 'val': {seed: [] for seed in self.seed_dirs}}
        pcas = {}
        for i, seed_dir in enumerate(self.seed_dirs):
            agg_dir = seed_dir / Path('aggregations')
            representations['train'][seed_dir] = np.load(agg_dir / 'trainrepresentations.npy', allow_pickle=True)
            representations['val'][seed_dir] = np.load(agg_dir / 'valrepresentations.npy', allow_pickle=True)
            activations['train'][seed_dir] = np.load(agg_dir / 'trainactivations.npy', allow_pickle=True)
            activations['val'][seed_dir] = np.load(agg_dir / 'valactivations.npy', allow_pickle=True)
            ys['train'][seed_dir] = np.load(agg_dir / 'trainy.npy', allow_pickle=True)
            ys['val'][seed_dir] = np.load(agg_dir / 'valy.npy', allow_pickle=True)
            pcas[seed_dir] = PCA(n_components=2, random_state=seeds[i])
            pcas[seed_dir].fit(representations['train'][seed_dir][-1])

        num_epochs = len(representations['train'][self.seed_dirs[0]])
        frames = []
        rows, columns = calc_rows_cols_reps(num_seeds)
        for epoch in range(num_epochs):
            var_frame = plot_gif_reps(
                    rows, columns, num_seeds, pcas, ys,
                    representations, epoch, self.seed_dirs)
            frames.append(var_frame)
        reps_dir = self.logdir / Path('representations')
        create_dir_safe(reps_dir)
        gif.save(frames,
                 reps_dir / Path('repstrain.gif'),
                 duration=100)
        print(f'Created representation trajectory gifs')    

    def visualize_parameters_fig(self, param_params):
        clss = param_params["num_classes"]
        for i, seed_dir in enumerate(self.seed_dirs):
            agg_dir = seed_dir / Path('aggregations')
            step_dict = torch.load(agg_dir / 'parameters.pt')
            if i == 0:
                parameter_dict = {
                    key: [step_dict[key].view(
                        step_dict[key].size(0), *return_shape_params(
                            key, self.embed_dim, self.num_classes)
                        )] for key in step_dict.keys()
                }
            else:
                for key in step_dict.keys():
                    params = step_dict[key]
                    params = params.view(
                        params.size(0), *return_shape_params(
                            key, self.embed_dim, self.num_classes))
                    parameter_dict[key].append(params)
        for key in parameter_dict.keys():
            parameter_dict[key] = torch.stack(parameter_dict[key], dim=1)

        rows, columns = calc_rows_cols(self.embed_dim)
        key = 'lin1.weight'
        params = parameter_dict[key]
        num_steps = params.size(0)
        steps_per_epoch = num_steps // self.num_epochs

        for (i, seed) in enumerate(self.seeds):
            vmin = params[:, i].min()
            vmax = params[:, i].max()
            frames = []
                    # Create a frame every 4 times an epoch
            for j in range(0, num_steps, steps_per_epoch // 4):
                frame = plot_gif_parameters_fig(rows, columns, self.embed_dim, params[j, i], vmin, vmax)
                frames.append(frame)
                #params_3d.append(params[j, i].reshape(784, -1))
            parameter_dir = self.logdir / Path('parameters')
            create_dir_safe(parameter_dir)
            gif.save(frames,
                    parameter_dir / Path(f'params_{key}_seed{seed}_clss{clss}_fig.gif'),
                    duration=100)

        print(f'Created parameter (first layer) trajectory figure gifs')

    @gif.frame
    def parameter_plot(self, data, i):
        rows, columns = self.calc_rows_columns()
        fig, axes = plt.subplots(rows, columns, figsize=(10, 10), dpi=100)
        for i in range(rows):
            for j in range(columns):
                embed_ix = (i * columns + j)
                if embed_ix < self.embed_dim:
                    axes[i, j].imshow(data[:, :, embed_ix])
        plt.tight_layout()
        parameter_dir = self.logdir / Path('parameters')
        create_dir_safe(parameter_dir)
        plt.savefig(parameter_dir / f'{i}')
        return fig

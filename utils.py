import gif
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment


def l1_norm(w, lam):
    return lam * w.abs().sum()

def l2_norm(w, lam):
    return lam * torch.linalg.norm(w, ord=2) #y.pow(2).sum().sqrt()

def calc_rows_cols(size):
    rows = int(np.sqrt(size))
    if size % rows == 0:
        return rows, rows
    else:
        columns = size // rows
        rows += 1
        return rows, columns

def calc_rows_cols_reps(size):
    rows = int(np.sqrt(size))
    if size / rows == rows:
        return rows, rows
    else:
        columns = size // rows
        if columns*rows < size:
            rows += 1
        return rows, columns

def create_dir_safe(p):
    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)


@gif.frame
def plot_first_layer(rows, columns, embed_dim, data, vmin, vmax):
    fig, axes = plt.subplots(rows, columns, figsize=(7, 7), dpi=100)
    for i in range(rows):
        for j in range(columns):
            embed_ix = (i * columns + j)
            if embed_ix < embed_dim:
                axes[i, j].imshow(data[:, :, embed_ix],
                                  vmin=vmin,
                                  vmax=vmax)
    plt.tight_layout()


@gif.frame
def plot_representations(rows, columns, num_seed, pcas, ys, representations, epoch, seed_dirs):
    fig, axes = plt.subplots(rows, columns, figsize=(10, 10), dpi=100)
    cmap = cm.get_cmap('tab20')
    for i in range(rows):
        for j in range(columns):
            seed_ix = (i * columns + j)
            if seed_ix < num_seed:
                X = pcas[seed_dirs[seed_ix]].transform(representations['train'][seed_dirs[seed_ix]][epoch])
                if rows > 1 and columns > 1:
                    for lab in range(10):
                        indices = ys['train'][seed_dirs[seed_ix]][epoch]==lab
                        axes[i,j].scatter(X[indices,0], X[indices,1], c=np.array(cmap(lab)).reshape(1,4), alpha=0.8)
                    axes[i,j].set_xlim([-10,10])
                    axes[i,j].set_ylim([-10,10])
                else:
                    for lab in range(10):
                        indices = ys['train'][seed_dirs[seed_ix]][epoch]==lab
                        axes[j].scatter(X[indices,0], X[indices,1], c=np.array(cmap(lab)).reshape(1,4), alpha=0.8)
                    axes[j].set_xlim([-10,10])
                    axes[j].set_ylim([-10,10])
    plt.tight_layout()


@gif.frame
def plot_bias(data, vmin, vmax):
    fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=100)
    ax.imshow(data.view(1, data.size(0)), vmin=vmin, vmax=vmax)


@gif.frame
def plot_deep_layer(data, vmin, vmax):
    fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=100)
    ax.imshow(data, vmin=vmin, vmax=vmax)

# ---- Pixels
def plot_gif_reps(rows, columns, num_seed, pcas, ys, representations, epoch, seed_dirs):
    frame = plot_representations(rows, columns, num_seed, pcas, ys, representations, epoch, seed_dirs)
    return frame


def plot_gif_parameters(rows, columns, embed_dim, data, vmin, vmax, key):
    if key == 'lin1.weight':
        frame = plot_first_layer(rows, columns, embed_dim, data, vmin, vmax)
    elif key == 'lin1.bias' or key == 'lin2.bias':
        frame = plot_bias(data, vmin, vmax)
    elif key == 'lin2.weight':
        frame = plot_deep_layer(data, vmin, vmax)
    return frame


def return_shape_params(key, embed_dim, num_classes):
    if key == 'lin1.weight':
        return (28, 28, embed_dim)
    elif key == 'lin1.bias':
        return (embed_dim, )
    elif key == 'lin2.weight':
        return (embed_dim, num_classes)
    elif key == 'lin2.bias':
        return (num_classes, )

@gif.frame
def plot_first_layer_fig(rows, columns, embed_dim, data, vmin, vmax):
    fig, axes = plt.subplots(rows, columns, figsize=(7, 7), dpi=100)
    data = data.reshape(784, -1)
    data, indices = torch.sort(data, 0)
    x = np.linspace(1, 784, 784)
    for i in range(rows):
        for j in range(columns):
            embed_ix = (i * columns + j)
            if embed_ix < embed_dim:                
                ar = data[:, embed_ix]
                ar = np.round(ar, decimals=3)
                unique, counts = np.unique(ar, return_counts=True)
                ar = np.asarray((unique, counts)).T
                axes[i, j].plot(ar[:,0], ar[:,1])
                axes[i, j].set_xlim([vmin, vmax])
    plt.tight_layout()

def plot_gif_parameters_fig(rows, columns, embed_dim, data, vmin, vmax):
    frame = plot_first_layer_fig(rows, columns, embed_dim, data, vmin, vmax)
    return frame

def calculate_permutation_correlation(
    seed_dirs, seeds, embed_dim, num_classes, viz_dir):
    df = pd.DataFrame(np.zeros((len(seed_dirs) - 1, 3)),
                      columns=['Corr-W1', 'Corr-W2', 'Permutation percentage'],
                      index=seeds[1:])
    seed_final_params = []
    for seed_dir in seed_dirs:
        parameter_p = seed_dir / Path('aggregations/parameters.pt')
        parameters = torch.load(parameter_p)
        param_dict = {}
        for (key, val) in parameters.items():
            param_dict[key] = val[-1]
        seed_final_params.append(param_dict)
    seed_one = seed_final_params[0]
    ixs = []
    seed_corr = {'lin1.weight': [],
                 'lin2.weight': []}
    perm_perc = []
    flag = True
    for (s, seed) in enumerate(seed_final_params[1:]):
        param_corr = {'lin1.weight': np.empty((embed_dim, embed_dim)),
                      'lin2.weight': np.empty((embed_dim, embed_dim))}
        for i in range(embed_dim):
            for j in range(embed_dim):
                r, p = pearsonr(seed_one['lin1.weight'][i], seed['lin1.weight'][j])
                param_corr['lin1.weight'][i, j] = r

        row_ix1, col_ix1 = linear_sum_assignment(-np.abs(param_corr['lin1.weight']))
        if flag:
            ixs.append(row_ix1)
            ixs.append(col_ix1)
            flag=False
        else:
            ixs.append(col_ix1)
        for i in range(embed_dim):
            for j in range(embed_dim):
                r, p = pearsonr(seed_one['lin2.weight'][:, i], seed['lin2.weight'][:, j])
                param_corr['lin2.weight'][i, j] = r
        row_ix2, col_ix2 = linear_sum_assignment(-np.abs(param_corr['lin2.weight']))
        df.loc[seeds[s + 1], 'Corr-W1'] = np.abs(param_corr['lin1.weight'][row_ix1, col_ix1]).mean()
        df.loc[seeds[s + 1], 'Corr-W2'] = np.abs(param_corr['lin2.weight'][row_ix2, col_ix2]).mean()
        df.loc[seeds[s + 1], 'Permutation percentage'] = (col_ix1 == col_ix2).sum() / embed_dim
    df.to_csv(viz_dir / 'permutation_results.csv')
    print(f'Num classes: {num_classes}')
    print(df)

def get_hyperparameters(args):
    lambdas = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.05]
    if len(args) > 1:
        n = int(args[1])
        hyperparams = {
            "classes": [n // 6 + 2],
            "loss_func": [{ "func": l1_norm, "lambda": lambdas[n % 6]}],
            "optimizers": [{
                "name": "SGD",
                "epochs": 50,
                "params": {
                    "lr": 1E-2,
                    "weight_decay": 0
                }
            }]
        }
    else:
        hyperparams = {
            "classes": [2],
            "loss_func": [
                { "func": l1_norm, "lambda": 0.001}
            ],
            "optimizers": [{
                "name": "SGD",
                "epochs": 50,
                "params": {
                    "lr": 1E-2,
                    "weight_decay": 0
                }
            }]
        }
    return hyperparams

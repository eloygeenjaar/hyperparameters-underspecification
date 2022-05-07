from gc import set_debug
from random import seed
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

np.set_printoptions(2, suppress=True)
embed_dim = 16
logs = list(Path('./logs').iterdir())

for num_classes in range(2, 11):
    seeds = [0, 42, 1337, 3213, 9999, 5783, 1023]
    logs = [f'./logs/model-{seed}-ed16-lr0.01-nc{num_classes}' for seed in seeds]
    seed_final_params = []
    for log in logs:
        parameter_p = log / Path('aggregations/parameters.pt')
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
    for seed in seed_final_params[1:]:
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
        seed_corr['lin1.weight'].append(param_corr['lin1.weight'][row_ix1, col_ix1].mean())
        for i in range(embed_dim):
            for j in range(embed_dim):
                r, p = pearsonr(seed_one['lin2.weight'][:, i], seed['lin2.weight'][:, j])
                param_corr['lin2.weight'][i, j] = r
        row_ix2, col_ix2 = linear_sum_assignment(-np.abs(param_corr['lin2.weight']))
        seed_corr['lin2.weight'].append(param_corr['lin2.weight'][row_ix2, col_ix2].mean())
        perm_perc.append((col_ix1 == col_ix2).sum() / embed_dim)
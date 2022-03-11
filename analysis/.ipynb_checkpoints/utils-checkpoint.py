"""
Various functions used in the analysis scripts
"""

import mgplvm
import numpy as np
import pickle
import torch
from torch import optim
from sklearn import decomposition

def init_cv(Y, T1, d, fa = True, scale = None):
    n_samples = Y.shape[0]
    mu = np.random.normal(0, 1, (n_samples, Y.shape[-1], d)) * 0.1
    if fa:
        pca = decomposition.FactorAnalysis(n_components=d)
        for j in range(n_samples):
            mu[j, T1, :] = pca.fit_transform(Y[j][:, T1].T)
            mu[j, T1, :] = 0.5 * mu[j, T1, :] / np.std(mu[j, T1, :], axis=0, keepdims=True)
            if scale is not None:
                mu[j, T1, :] = mu[j, T1, :] * scale / (np.amax(mu[j, T1, :]) - np.amin(mu[j, T1, :]))
    return mu


def not_in(arr, inds):
    mask = np.ones(arr.size, dtype=bool)
    mask[inds] = False
    return arr[mask]


def get_init(type_, d):
    if type_ in ['linear', 'Linear']:
        return 'fa'
    else:
        if d == 1:
            return 'random'
        else:
            return 'fa'


def comp_var_exp(mod, Y, exp=False):
    query = mod.lat_dist.prms[0].transpose(-1, -2)
    Ypred = detach(mod.svgp.predict(query[None, ...], False)[0])
    if exp: Ypred = np.exp(Ypred)
    var_cap = 1 - np.var(Y - Ypred[0, ...]) / np.var(Y)
    return var_cap

def predict(mod, exp=False):
    query = mod.lat_dist.prms[0].transpose(-1, -2)
    Ypred = detach(mod.svgp.predict(query[None, ...], False)[0])
    if exp: Ypred = np.exp(Ypred)
    return Ypred[0, ...]


def basedir():
    return './'


def detach(tensor):
    return tensor.detach().cpu().numpy()



"""
Run the synthetic cross-validation analyses in figure 2.
Use: 'python synthetic_fit.py (True/False)'
The optional command line argument indicates whether to compute training likelihoods (panel a; default) or cross-validated errors (panel b).
"""

import mgplvm as mgp
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import sys
from utils import detach, basedir, not_in
from sklearn.decomposition import FactorAnalysis
device = mgp.utils.get_device()

np.random.seed(11131401)
torch.manual_seed(11131401)

mixedlab = '_mixed'

print(sys.argv)

if len(sys.argv) > 1:
    cv = (sys.argv[1] in ['true', 'True', 'TRUE'])
else:
    cv = False
print('cv:', cv)


T = 240
n = 40
ts = np.arange(T)
fit_ts = torch.tensor(ts)[None, None, :].to(device)
dts_2 = (ts[:, None] - ts[None, :])**2

####generate synthetic data####
d_true = 3
xs_true = []
ells = [9,10,11]
for d in range(d_true):
    K = np.exp(-dts_2/(2*ells[d]**2)) # TxT
    L = np.linalg.cholesky(K + np.eye(T)*1e-6) # TxT
    xs_true.append((L @ np.random.normal(0, 1, (T, 1))).T) # DxT
xs_true = np.concatenate(xs_true, axis = 0)
C_true = np.random.normal(0, 1, (n, d_true))
C_true *= np.array([[1.0, 1.0, 0.80]])
    
F_true = C_true @ xs_true # n x T
sig = 0.6*np.std(F_true)


noises = np.random.normal(0, sig, (F_true.shape)) #n x T
Y = F_true + noises
print('noise std:', sig)
Ymean = np.mean(Y, axis = -1, keepdims = True)
Y = Y - Ymean
Y = Y[None, ...] #1 x n x T
n_samples = 1
nreps = 10

### initialize arrays for storing data ####
ds_fit = np.arange(1, 11)
LLs = np.zeros((nreps, len(ds_fit), 4))
norm_MSEs = np.zeros((nreps, len(ds_fit), 4))
MSEs = np.zeros((nreps, len(ds_fit), 4))
LL_trains = np.zeros((nreps, len(ds_fit), 4))

### split into train and test ####
N1 = np.arange(n)[:20]
T1 = np.arange(T)[:int(round(2*T/4))]

if cv:
    savename_cv = 'data/cv_data_nrep'+str(nreps)+'_dmax'+str(ds_fit[-1])+mixedlab
    T2, N2 = not_in(np.arange(T), T1), not_in(np.arange(n), N1)
    low_bound = (F_true-Ymean)[N2, :][:, T2] - Y[0, N2, :][:, T2]
    low_bound = np.mean(low_bound**2)
    print('lower bound:', low_bound, 'var:', np.var(Y[0, N2, :][:, T2]))
    Ytrain = Y[..., T1]
else:
    Y = Y[..., T1]
    Ytrain = Y
    n_samples, n, T = Y.shape
    ts = np.arange(T)
    fit_ts = torch.tensor(ts)[None, None, :].to(device)
    savename_LL = 'data/LLs_nrep'+str(nreps)+'_dmax'+str(ds_fit[-1])+mixedlab
    
data = torch.tensor(Y).to(device)
def cb_ard(mod, i, loss):
    if i % 400 == 0:
        print('')
        ls, ss = detach(mod.obs.dim_scale).flatten()**(-1), np.mean(detach(mod.lat_dist.scale), axis = (0, -1))
        ms = np.sqrt(np.mean(detach(mod.lat_dist.nu)**2, axis = (0, -1)))
        args = np.argsort(ls)
        print(np.round(ls[args], 2), '', np.round(ss[args], 2), '', np.round(ms[args], 2))
    return False
        
print(Y.shape)

for nrep in range(nreps):
    print('\n\nnrep:', nrep)
    for nmod in range(0,4):
        iter_ = iter([ds_fit[-1]]) if nmod == 3 else iter(ds_fit)
        for dnum, d_fit in enumerate(iter_):

            manif = mgp.manifolds.Euclid(T, d_fit)

            ### instantiate model ###
            if nmod == 0: #FA
                fa = FactorAnalysis(n_components=d_fit)
                fa.fit_transform(Ytrain[0, ...])
                mutrain = fa.components_.T #initialize from training latents (ntrial x mtrain x d)
                mu = np.concatenate([mutrain, np.zeros((Y.shape[-1] - Ytrain.shape[-1], d_fit))])[None, ...]
                lprior = mgp.lpriors.Uniform(manif)
                lat_dist = mgp.rdist.ReLie(manif, T, n_samples, sigma = 0.2, diagonal=True, initialization = 'fa', Y=Y, mu = mu)
                mod = mgp.models.Lgplvm(n, T, d_fit, n_samples, lat_dist, lprior, Y = Ytrain, Bayesian = False).to(device)

            elif nmod == 1: #GPFA (variational)
                lprior = mgp.lpriors.Null(manif)
                lat_dist = mgp.rdist.GP_circ(manif, T, n_samples, fit_ts, _scale=1.0, ell = 8)
                mod = mgp.models.Lgplvm(n, T, d_fit, n_samples, lat_dist, lprior, Y = Ytrain, Bayesian = False).to(device)

            elif nmod in [2,3]: #bGPFA
                lprior = mgp.lpriors.Null(manif)
                lat_dist = mgp.rdist.GP_circ(manif, T, n_samples, fit_ts, _scale=1.0, ell = 8)
                lik = mgp.likelihoods.Gaussian(n, Y=Y, d=d_fit)

                if nmod == 2: #not ard
                    mod = mgp.models.Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, lik, Y = Ytrain, learn_scale = True).to(device)
                else: #ard
                    mod = mgp.models.Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, lik, Y = Ytrain, learn_scale = False, ard = True, rel_scale = 0.5).to(device)

            cb, nmax = (cb_ard, 2501) if nmod == 3 else (None, 2501)
            n_mc = 20
            
            ### set parameters for training ####
            train_ps = mgp.crossval.training_params(max_steps = nmax, n_mc = n_mc, burnin = 50, lrate = 5e-2, callback = cb)

            if cv: #cross-validated
                mod, split = mgp.crossval.train_cv(mod, Y, device, train_ps, T1 = T1, N1 = N1, test = False)
                if nrep == 0 and nmod == 3:
                    torch.save(mod, savename_cv+'_bGPFA.pt')#save ARD model for further analysis
                
                MSE, LL, var_cap, norm_MSE = mgp.crossval.test_cv(mod, split, device, n_mc = 1000, sample_mean = False, sample_X = False)
                LLs[nrep, dnum, nmod] = LL
                norm_MSEs[nrep, dnum, nmod] = norm_MSE
                MSEs[nrep, dnum, nmod] = MSE
                
                svgp_elbo, kl = mod(data, 1000, m=T, analytic_kl=(False if nmod == 0 else True), neuron_idxs = N1)
                LL_train = (svgp_elbo-kl).item()/np.prod(Y.shape)*n/len(N1)
                LL_trains[nrep, dnum, nmod] = LL_train
                print(d_fit, 'LL:', np.round(LL, 4), 'R2:', np.round(var_cap, 4), 'MSE:', np.round(MSE, 4), 'norm MSE:', np.round(norm_MSE, 4), 'train LL:', np.round(LL_train, 4))
            else: #training error
                mod_train = mgp.crossval.train_model(mod, data, train_ps)
                svgp_elbo, kl = mod(data, 1000, m=T, analytic_kl=(False if nmod == 0 else True))
                LL = (svgp_elbo-kl).item()/np.prod(Y.shape)
                LLs[nrep, dnum, nmod] = LL
                print(d_fit, nmod, LL)
                
    save = True
    if cv and save:
        pickle.dump([ds_fit, LLs, norm_MSEs, MSEs, LL_trains], open(savename_cv+'_temp.pickled', 'wb'))
    elif save:
        pickle.dump([ds_fit, LLs], open(savename_LL+'_temp.pickled', 'wb'))

if cv and save:
    pickle.dump([ds_fit, LLs, norm_MSEs, MSEs, LL_trains], open(savename_cv+'.pickled', 'wb'))
elif save:
    pickle.dump([ds_fit, LLs], open(savename_LL+'.pickled', 'wb'))

        
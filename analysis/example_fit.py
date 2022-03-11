"""
fit the example models used for analyses in figure 2c-e
use: python example_fit.py
"""

import mgplvm as mgp
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import detach, basedir, not_in
device = mgp.utils.get_device()

np.random.seed(11121401)
torch.manual_seed(11121401)

def cb_ard(mod, i, loss):
    if i % 150 == 0:
        print('')
        ls, ss = detach(mod.obs.dim_scale).flatten()**(-1), np.mean(detach(mod.lat_dist.scale), axis = (0, -1))
        ms = np.sqrt(np.mean(detach(mod.lat_dist.nu)**2, axis = (0, -1)))
        args = np.argsort(ls)
        print(np.round(ls[args], 2), '', np.round(ss[args], 2), '', np.round(ms[args], 2))
    return False


#### generate latents and tuning curves ####
n, T, n_samples = 35, 180, 1
ts = np.arange(T)
fit_ts = torch.tensor(ts)[None, None, :].to(device)
dts_2 = (ts[:, None] - ts[None, :])**2
ell = 20
d_true = 2
K = np.exp(-dts_2/(2*ell**2)) # TxT
L = np.linalg.cholesky(K + np.eye(T)*1e-6) # TxT
xs = (L @ np.random.normal(0, 1, (T, d_true))).T # DxT
C = np.random.normal(0, 1, (n, d_true))*0.5
F = C @ xs # n x T

#### draw noise from Gaussian, Poisson and NegBinomial model ####
sig = 1.5*np.std(F)
YGauss = F + np.random.normal(0, sig, (F.shape)) #n x T

c_p = 0
YPois = np.random.poisson(np.exp(F + c_p)).astype(float)

c_nb = -1.55
p_nb = np.exp(F+c_nb)/(1+np.exp(F+c_nb))
r_nb = np.random.uniform(1, 10, n)##n failues
#numpy defines in terms of #successes so we substitute 1 -> 1-p

print(r_nb.shape, p_nb.shape)
YNB = np.random.negative_binomial(r_nb, 1-p_nb.T).astype(float).T


print(np.mean(YPois, axis = -1))
print(np.amax(YPois, axis = -1))
print(np.mean(YNB, axis = -1))
print(np.amax(YNB, axis = -1))
print(np.mean(np.mean(YNB, axis = -1) / np.mean(YPois, axis = -1)))

mods = []

### train models ####
d_fit = 10
labs = ['Gauss', 'Pois', 'NB']
for i, newY in enumerate([YGauss, YPois, YNB]):
    print('\n', i, labs[i])
    Y = newY[None, ...]
    data = torch.tensor(Y).to(device)

    if i == 0:
        lik = mgp.likelihoods.Gaussian(n, Y=Y, d=d_fit)
    elif i == 1:
        lik = mgp.likelihoods.Poisson(n)
    elif i == 2:
        lik = mgp.likelihoods.NegativeBinomial(n, Y=Y)
    rel_scale = 0.1 if i == 1 else 1 #smaller initialization for Poisson due to exponential nonlinearity

    manif = mgp.manifolds.Euclid(T, d_fit)
    lprior = mgp.lpriors.Null(manif)
    lat_dist = mgp.rdist.GP_circ(manif, T, n_samples, fit_ts, _scale=1, ell = 20*0.8)
    mod = mgp.models.Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, lik, Y = Y, learn_scale = False, ard = True, rel_scale = rel_scale).to(device)
        
    train_ps = mgp.crossval.training_params(max_steps = 2501, n_mc = 10, burnin = 50, lrate = 5e-2, callback = cb_ard)
    mod_train = mgp.crossval.train_model(mod, data, train_ps)
    LL = mod.calc_LL(data, 100)
    print(i, LL)
    mods.append(mod)
    torch.save(mod, 'data/example_'+labs[i]+'.pt')
    
    
data = {'YGauss': YGauss,
       'YPoiss': YPois,
        'YNB': YNB,
        'c_p': c_p,
        'c_nb': c_nb,
        'r_nb': r_nb,
        'p_nb': p_nb,
        'xs': xs,
        'C': C,
        'F': F,
        'ell': ell
       }
pickle.dump(data, open('data/example_data_fits.pickled', 'wb'))
        
    
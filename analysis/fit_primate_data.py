"""
code for fitting models to the primate data
use: python fit_primate_data.py M1/S1/both true/false (_rep*)
first argument is the brain region to fit
second argument is whether or not to shift M1 spikes by 100 ms
third (optional) argument sets a new seed and writes to a different filename.
"""

import numpy as np
import matplotlib.pyplot as plt
import mgplvm as mgp
import torch
import time
import pickle
from scipy.stats import binned_statistic
import sys
from utils import basedir, detach
from load_joey import load_data
torch.set_default_dtype(torch.float64)
device = mgp.utils.get_device("cuda")  # get_device("cpu")
np.random.seed(11101401)
torch.manual_seed(11101401)

region = sys.argv[1] #M1/S1/both
shift = sys.argv[2] in ['True', 'true', 'TRUE'] #true/false
if len(sys.argv) > 3:
    lab = sys.argv[3] #rep*
else:
    lab = ''


if lab[1:4] == 'rep': #new model with new seed
    print('changing seed!')
    np.random.seed(11101401+int(lab[-1])-1)
    torch.manual_seed(11101401+int(lab[-1])-1)

##### data parameters ######
name = 'indy_20160426_01'
shiftsize = 100 #in ms

### model parameters ###
likelihood = 'NegBinom'
circ = True
n_mc = 10 #number of monte carlo samples
lrate = 5e-2
max_steps = 2501
_scale = 0.9
    
binsize = 25
d_fit = 25

batch_size = 1750 ###reduce this if not enough memory###
rel_scale = 3 #rho

#### load data ####
Y = load_data(name = name, region = region, binsize = binsize, behavior = False, shift = shift, thresh = 2, shiftsize = shiftsize)

print('Y shape:', Y.shape, 'batch size:', batch_size, 'label:', lab, 'dfit:', d_fit, 'rel_scale:', rel_scale)
n_samples, n, T = Y.shape
if shift:
    print("shifting!", shiftsize)
    shiftlab = str(shiftsize)
else:
    print("not shifting!")
    shiftlab = 'none'

### progress function ###
t0 = time.time()
def cb_ard(mod, i, loss):
    if i % 20 == 0:
        print('')
        global t0
        ls, ss = detach(mod.obs.dim_scale).flatten()**(-1), np.mean(detach(mod.lat_dist.scale), axis = (0, -1))
        lambdas = ls**(-2) #compute participation ratio
        dim = np.sum(lambdas)**2 / np.sum(lambdas**2) 
        ms = np.sqrt(np.mean(detach(mod.lat_dist.nu)**2, axis = (0, -1)))
        args = np.argsort(ls)
        print(region, shiftlab, dim, np.round(ls[args], 2), '', np.round(ss[args], 2), '', np.round(ms[args], 2), np.round(time.time()-t0))
        t0 = time.time()
    return False
    
#### construct model ####
data = torch.tensor(Y).to(device)
fit_ts = torch.tensor(np.arange(T))[None, None, :]
manif = mgp.manifolds.Euclid(T, d_fit)
lprior = mgp.lpriors.Null(manif)

if circ: #circuland parameterization
    lat_dist = mgp.rdist.GP_circ(manif, T, n_samples, fit_ts, _scale=_scale, ell = 200/binsize) #initial ell ~200ms
else: #diagonal parameterization
    lat_dist = mgp.rdist.GP_diag(manif, T, n_samples, fit_ts, _scale=_scale, ell = 200/binsize) #initial ell ~200ms

lik = mgp.likelihoods.NegativeBinomial(n, Y=Y)
mod = mgp.models.Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, lik, ard = True, learn_scale = False, Y = Y, rel_scale = rel_scale).to(device)

#### fit model ####
train_ps = mgp.crossval.training_params(max_steps = max_steps, n_mc = n_mc, burnin = 50, lrate = lrate, batch_size = batch_size, print_every = 10, callback = cb_ard, analytic_kl = True)
mod_train = mgp.crossval.train_model(mod, data, train_ps)


#### store model, latents, and predictions ####
print('\n')
fname = likelihood+'_d'+str(d_fit)+'_'+region+'_b'+str(binsize)
if shift:
    fname += '_shift'+str(shiftsize)
if circ:
    fname += '_circ'
fname += lab #can add a label

### store model and latents ###
torch.save(mod, 'data/'+fname+'.pt')
lats = mod.lat_dist.lat_mu.detach().cpu().numpy()  #latent means (ntrial, m, d)
pickle.dump([lats, Y], open('data/'+fname+'_lats_Y.pickled', 'wb'))


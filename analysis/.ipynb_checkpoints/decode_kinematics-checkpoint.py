"""
This code was used to fit the kinematic decoding models
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from utils import basedir, not_in
import glmnet_python
import scipy
from glmnet import glmnet
from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet
from glmnetPrint import glmnetPrint
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPredict import cvglmnetPredict
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr
import torch
import sys
import mgplvm as mgp
from load_joey import load_data

region = sys.argv[1] #M1/S1/both
shift = sys.argv[2] in ['True', 'true', 'TRUE'] #true/false
if len(sys.argv) > 3:
    lab = sys.argv[3] #rep*
else:
    lab = ''


def zscore(x, ax = 1):
    x = (x - np.mean(x, axis = ax, keepdims = True)) / np.std(x, axis = ax, keepdims = True)
    return x


#### fitting parameters ####
alpha = 0
deriv = 1 #which derivative of position to predict (velocity has best performance)
spline = True #fit cubic spline to behavior
nfold = 5 #n-fold cross-validation
nreps = 10 #number of splits to try
delays = np.arange(-150, 311, 20)

def decode(x, y, lab = ''):
    all_vars_sep = np.zeros( (nreps, len(delays), nfold) ) #store variance explained
    all_vars = np.zeros( (nreps, len(delays), nfold) ) #store variance explained
    for ndelay, delay in enumerate(delays):
        np.random.seed(13583101) #same split for each delay
        scipy.random.seed(13583101)
        if spline:
            ts = np.arange(y.shape[0])*binsize #measured in ms
            cs = CubicSpline(ts, y)
            newy = cs(ts+delay, deriv)
            newx = x.copy()
        else:
            for _ in range(deriv):
                newx = x[1:, ...].copy()
                newy = y[1:, ...] - y[:-1, ...]
        print(newx.shape, newy.shape)

        for nrep in range(nreps):
            inds = np.arange(y.shape[0])
            np.random.shuffle(inds) #random partition each time
            sfold = int(np.floor(len(inds)/nfold))
            splits_test = [inds[i*sfold : (i+1)*sfold] for i in range(nfold)]

            for n in range(nfold): #for each fold
                inds_train, inds_test = np.ones(len(inds), dtype=bool), splits_test[n]
                inds_train[inds_test] = False

                #first we do n-fold cross-validation on the held out data to set the regularization strength
                cvfits = [cvglmnet(x = newx[inds_train, :], y = newy[inds_train, i], ptype = 'mse', nfolds = nfold, alpha = alpha) for i in range(2)]

                #then we use the model trained on the full training data with the inferred regularization strength to predict on the test data
                preds = np.array([cvglmnetPredict(cvfit, newx = newx[inds_test, ...], s='lambda_min').flatten() for cvfit in cvfits]).T

                testy = newy[inds_test, :] #true output
                RMSE = np.sqrt(((preds-testy)**2).mean()/np.var(testy)) #RMSE
                #var_old = 1 - ((preds-testy)**2).mean()/np.var(testy) #R^2
                var = pearsonr(preds.flatten(), testy.flatten())[0]**2
                var_sep = np.mean([pearsonr(preds[:, i], testy[:, i])[0]**2 for i in range(2)])
                all_vars[nrep, ndelay, n] = var #store R^2
                all_vars_sep[nrep, ndelay, n] = var_sep #store R^2
            if nrep % 2 == 0:
                print(nrep, lab, region, likelihood, delay, np.mean(all_vars[nrep, ndelay]), np.mean(all_vars_sep[nrep, ndelay]))
            
    return all_vars, all_vars_sep


#### data parameters ####
likelihood = 'NegBinom'
shiftsize = 100
binsize = 25
circ = True
move = True
d_fit = 25
fname = likelihood+'_d'+str(d_fit)+'_'+region+'_b'+str(binsize)
if shift:
    fname += '_shift'+str(shiftsize)
if circ:
    fname += '_circ'
fname += lab

print('label:', lab)

#load data without and with convolution
Y, locs, targets = load_data(region = region, binsize = binsize, behavior = True, shift = shift, shiftsize = shiftsize,
                            convolve = False, lens = 50)
Yc, _, _ = load_data(region = region, binsize = binsize, behavior = True, shift = shift, shiftsize = shiftsize,
                            convolve = True, lens = 50)


#### generate predicted activity ####
print('predicting activity')
device = mgp.utils.get_device()
mod = torch.load('data/'+fname+'.pt', map_location={'cuda:1':'cuda:0'}).to(device)
for p in mod.parameters(): mod.requires_grad = False
#### compute latent mean ###
lats = mod.lat_dist.lat_mu

query = torch.tensor(lats).transpose(-1, -2).to(device)  #(ntrial, d, m)
print(query.shape)
Ypreds = []

T = lats.shape[-2]
step = int(np.round(T/10))
for i in range(100):
    Ys = []
    for s in range(9):
        if s == 8:
            newq = query[..., s*step:]
        else:
            newq = query[..., s*step:(s+1)*step]
        Ypred = mod.svgp.sample(newq, n_mc=10, noise=False)
        Ypred = Ypred.detach().mean(0).cpu().numpy()  #(ntrial x N x m)
        Ys.append(Ypred)
    Ypreds.append(np.concatenate(Ys, axis = -1))

Ypred = np.array(Ypreds)
print(Ypred.shape)
Ypred = np.mean(Ypred, axis = 0)
print(Ypred.shape)
x = Ypred[0, ...]

print('Y:', Y.shape)
query = None
mod = None
torch.cuda.empty_cache()
    
if move: #only fit to the period with actual behavior
    move_arg = int(1430/binsize*1000)
    locs = locs[:move_arg, :]
    x = x[:, :move_arg]
    Y = Y[..., :move_arg]
    Yc = Yc[..., :move_arg]
    
y = locs.T
x = zscore(x).T
y = zscore(y).T
print(x.shape, y.shape, Y.shape, Yc.shape)

all_vars, all_vars_sep = decode(x, y, lab = '')

fit_ctrl = True #determine whether to also fit to raw data
if fit_ctrl:
    x_con = Yc[0, ...] #convolved with 50ms kernel
    x_con = zscore(x_con).T
    all_vars_con, all_vars_con_sep = decode(x_con, y, lab = 'con')

    x_raw = Y[0, ...]
    x_raw = zscore(x_raw).T
    all_vars_raw, all_vars_raw_sep = decode(x_raw, y, lab = 'raw')
    
else:
    all_vars_con, all_vars_con_sep = [], []
    all_vars_raw, all_vars_raw_sep = [], []

data = {'delays': delays,
       'bgpfa': all_vars,
       'bgpfa_sep': all_vars_sep,
       'convolved': all_vars_con,
       'convolved_sep': all_vars_con_sep,
       'raw': all_vars_raw,
       'raw_sep': all_vars_raw_sep}
       
if move:
    fname += '_move'
pickle.dump(data, open('data/'+fname+'_kinematic_predictions.pickled', 'wb'))



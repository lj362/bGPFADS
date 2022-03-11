import numpy as np
import time
import torch
import mgplvm as mgp
import pickle
import copy
import mgplvm as mgp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, ttest_ind
from sklearn import decomposition
from utils import detach, basedir, not_in
from load_joey import load_data
device = mgp.utils.get_device()
torch.set_default_dtype(torch.float64)
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

#### fit a GP to the noisy data ####

binsize = 25
read_elbo = True
read_dims = True
d_fit = 25

labs = ['_rep'+str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
inds = np.arange(10)
    
shiftsize = 100
all_shifts = [0, 100]
shiftstr = ''
    
all_prs, all_naives = [[[] for _ in range(len(all_shifts))] for _ in range(2)]
#### consider dimensionality as a function of shift ####
for ishift, shift in enumerate(all_shifts):
    for lab in labs:
        if shift == 0:
            fname = 'NegBinom_d'+str(d_fit)+'_both_b'+str(binsize)+'_circ'+lab
        else:
            fname = 'NegBinom_d'+str(d_fit)+'_both_b'+str(binsize)+'_shift'+str(shift)+'_circ'+lab

        mod = torch.load('data/'+fname+'.pt', map_location={'cuda:1':'cuda:0'}).to(device)
        lambdas = detach(mod.obs.dim_scale).flatten()**2 #compute participation ratio
        naive = np.sum(np.log(np.sqrt(lambdas)) > -5)

        Y = load_data(name = 'indy_20160426_01', region = 'both', binsize = binsize, subsamp = False, behavior = False,
                  shift = (False if shift==0 else True), thresh = 2, shiftsize = shift)
        pca = decomposition.PCA()
        Ypca = np.sqrt(Y[0, ...].T)
        Ypca = pca.fit_transform(Ypca) #T x N
        pr = np.sum(pca.explained_variance_)**2 / np.sum(pca.explained_variance_**2) #classsical participation ratio
        all_prs[ishift].append(pr)
        all_naives[ishift].append(naive)

        print(shift, lab, naive, pr, '\n' )

all_data = [np.array(data) for data in all_naives]
print('dimensionality:', [[np.mean(data[inds]), np.std(data[inds])/np.sqrt(len(inds))] for data in all_data])
print(ttest_ind(all_data[0][inds], all_data[1][inds]), '\n')

all_elbos = [[] for _ in range(len(all_shifts))]
#### consider ELBO as a function of shift ####
for ishift, shift in enumerate(all_shifts):
    for lab in labs:
        if shift == 0:
            fname = 'NegBinom_d'+str(d_fit)+'_both_b'+str(binsize)+'_circ'+lab
        else:
            fname = 'NegBinom_d'+str(d_fit)+'_both_b'+str(binsize)+'_shift'+str(shift)+'_circ'+lab

        mod = torch.load('data/'+fname+'.pt', map_location={'cuda:1':'cuda:0'}).to(device)

        Y = load_data(name = 'indy_20160426_01', region = 'both', binsize = binsize, subsamp = False, behavior = False,
                  shift = (False if shift==0 else True), thresh = 2, shiftsize = shift)
        data = torch.tensor(Y).to(device)
        dataloader = mgp.optimisers.data.BatchDataLoader(data, batch_size=15000, batch_pool=None)
        for p in mod.parameters():
            p.requires_grad = False
        n_mc = 1
        elbos = []
        for nrep in range(50):
            if nrep > 0 and nrep % 10 == 0: print(nrep, np.mean(elbos[-10:])/np.prod(Y.shape))
            elbos_iter = []
            for sample_idxs, batch_idxs, batch in dataloader:
                svgp_elbo, kl = mod(batch,
                                      n_mc,
                                      batch_idxs=batch_idxs,
                                      sample_idxs=sample_idxs,
                                      neuron_idxs=None,
                                      m=Y.shape[-1],
                                      analytic_kl=True)
                elbo = (svgp_elbo-kl).item()
                #print(len(batch_idxs), elbo)
                elbos_iter.append( elbo*len(batch_idxs)/Y.shape[-1] )
            elbos.append(np.sum(elbos_iter))
        print(shift, lab, np.mean(elbos)/np.prod(Y.shape), '\n' )
        all_elbos[ishift].append(np.mean(elbos)/np.prod(Y.shape))


#### decoding performance
all_r2s = [[] for _ in range(len(all_shifts))]
for ishift, shift in enumerate(all_shifts):
    for lab in labs:
        if shift == 0:
            fname = 'NegBinom_d'+str(d_fit)+'_both_b'+str(binsize)+'_circ'+lab
        else:
            fname = 'NegBinom_d'+str(d_fit)+'_both_b'+str(binsize)+'_shift'+str(shift)+'_circ'+lab
        data = pickle.load(open('data/'+fname+'_move_kinematic_predictions.pickled', 'rb'))
        delays, r2s = data['delays'], data['bgpfa_sep']
        r2s =np.mean(r2s, axis = (0, -1)) #mean over splits and folds
        #print(delays, r2s)
        r2, delay = np.amax(r2s), delays[np.argmax(r2s)]
        print(shift, lab, r2, delay)
        all_r2s[ishift].append(r2)
        

cats = ['dims', 'elbos', 'r2s']
for i, all_data in enumerate([all_naives, all_elbos, all_r2s]):
    all_data = [np.array(data) for data in all_data]
    print('\n'+cats[i]+':', [[np.mean(data[inds]), np.std(data[inds])/np.sqrt(len(inds))] for data in all_data])
    print(ttest_ind(all_data[0][inds], all_data[1][inds]))


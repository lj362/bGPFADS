"""
code for analysing synthetic example fits and making preliminary plots
use: python plot_example_fit.py
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
write = False

data = pickle.load(open('data/example_data_fits.pickled', 'rb'))
xs = data['xs'].T
xs = xs - np.mean(xs, axis = 0, keepdims = True)

labs = ['Gauss', 'Pois', 'NB']
cols = ['b', 'g', 'r']
mods = []
all_lats = []
#### plot latents ####
plt.figure()
plt.plot(xs[:, 0], xs[:, 1], 'k-')
for i in range(3):
    mods.append(torch.load('data/example_'+labs[i]+'.pt'))
    lats = detach(mods[-1].lat_dist.lat_mu)[0, ...]
    dim_scales = detach(mods[-1].obs.dim_scale).flatten()
    inds = np.argsort(-dim_scales)[:2]
    lats = lats[..., inds]
    lats = lats - np.mean(lats, axis = 0, keepdims = True)
    print(lats.shape, xs.shape)
    
    #fit xs = mus @ T --> T = (mus' * mus)^(-1) * mus' * xs
    T = np.linalg.inv(lats.T @ lats) @ lats.T @ xs
    lats = lats @ T  #predicted values
    all_lats.append(lats)
    
    plt.plot(lats[:, 0], lats[:, 1], cols[i]+'-')
plt.legend(['Gaussian', 'Poisson', 'Negative Binomial'])
plt.savefig('figures/example_latents.png')
plt.close()

pickle.dump([xs]+all_lats, open('../figures/figure_data/synthetic/example_latents.pickled', 'wb'))

r_inf = detach(mods[-1].obs.likelihood.prms[0])
r_true = data['r_nb']
def plotfunc(r):
    return r
    return np.log(r**(-1))

plt.figure()
plt.scatter(plotfunc(r_true), plotfunc(r_inf), c = 'k', s = 20*np.mean(data['YNB'], axis = -1))

trues = plotfunc(np.array([np.amin(r_true), np.amax(r_true)]))
plt.plot(trues, trues, 'b-')
plt.xlabel('true r')
plt.ylabel('inferred r')
plt.savefig('figures/example_overdispersion.png')
plt.close()
pickle.dump([r_true, r_inf, np.mean(data['YNB'], axis = -1)], open('../figures/figure_data/synthetic/example_NB_overdispersion.pickled', 'wb'))

all_dim_scales, all_ms, all_ss = [], [], []
plt.figure()
ax = plt.axes(projection='3d')
for i in range(3):
    #kernel length scales
    _dim_scales = detach(mods[i].obs.dim_scale).flatten()
    all_dim_scales.append(_dim_scales)
    dim_scales = np.log(_dim_scales)
    #average over trials, time points
    ss = np.mean(detach(mods[i].lat_dist.scale), axis = (0, -1))
    ms = np.sqrt(np.mean(detach(mods[i].lat_dist.nu)**2, axis = (0, -1)))
    all_ss.append(ss)
    all_ms.append(ms)
    
    print('\n', labs[i], dim_scales, ms, ss)
    
    ax.scatter(dim_scales, ms, ss, c = cols[i], s = 30)
ax.set_xlabel(r'$\sigma_d$')
ax.set_ylabel(r'$||\mu||^2_2$')
ax.set_zlabel(r'$\langle s \rangle$')
ax.view_init(60, 120)
plt.legend(['Gaussian', 'Poisson', 'Negative Binomial'], frameon = False)
plt.savefig('figures/example_params.png')
plt.close()

pickle.dump(all_dim_scales, open('../figures/figure_data/synthetic/example_dim_scales.pickled', 'wb'))
pickle.dump(all_ms, open('../figures/figure_data/synthetic/example_lat_ms.pickled', 'wb'))
pickle.dump(all_ss, open('../figures/figure_data/synthetic/example_lat_ss.pickled', 'wb'))


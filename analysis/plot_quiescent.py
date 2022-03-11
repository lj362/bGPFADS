import numpy as np
import time
import torch
import mgplvm as mgp
import pickle
import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils import detach, basedir, not_in
from load_joey import load_data
from scipy.stats import pearsonr, binned_statistic
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn import decomposition
import sys
device = mgp.utils.get_device()
#device = torch.device("cuda:0")
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
write = True


##### load data ######
subsamp = False
region = 'M1'
binsize = 25
d_fit = 25
labstr = ''

#### load data and compute reach angles ####
Y, locs, targets = load_data(region = region, binsize = binsize, subsamp = subsamp, behavior = True, shift = False, shiftsize = 100)
ts = np.arange(locs.shape[0])*binsize/1000/60 #measured in minutes
cs = CubicSpline(ts*60, locs/1000)
print(np.amin(ts), np.amax(ts))
vels = cs(ts*60, 1) #first derivative of the spline
    
#### load model ####
fname = 'NegBinom_d'+str(d_fit)+'_'+region+'_b'+str(binsize)+'_circ'+labstr
mod = torch.load('data/'+fname+'.pt').to(device)
lats = detach(mod.lat_dist.lat_mu) #n_trials x m x d
scales = detach(mod.obs.dim_scale).flatten()
scale_args = np.argsort(-scales)
scales = scales[scale_args]
lats = lats[0, ...][:, scale_args] #sort by information content; T x d
ells = detach(mod.lat_dist.ell).flatten()[scale_args]

print('downsampling')
nsamp = 8
lats_sub = np.array([np.mean(lats[i*nsamp:(i+1)*nsamp, :], axis = 0) for i in range(int(np.floor(lats.shape[0]/nsamp)))])
print(lats_sub.shape)
#lats = lats[::6, :] #downsample to 100ms resolution; ~20k

print('computing differences')
diffs = lats_sub[None, ...] - lats_sub[:, None, :] # T x T x d
diffs = np.sqrt(np.sum(diffs**2, axis = -1))


RT_inds = pickle.load(open('data/RT_inds.pickled', 'rb'))
print(len(RT_inds), len(RT_inds[0]), np.concatenate(RT_inds).shape, ts.shape)

all_inds = np.concatenate(RT_inds)
print(all_inds[0], all_inds[-1])
ind_diffs = all_inds[1:] - all_inds[:-1]
print(np.amin(ind_diffs), np.amax(ind_diffs))
gap = np.argmax(ind_diffs)

###### lats sub #######
fig = plt.figure(figsize = (12, 2.5))
gs = fig.add_gridspec(1, 3, left=0.00, right=1.00, bottom=0.0, top=1.00, 
                      wspace = 0.35, hspace = 0.15, width_ratios = [1, 0.6, 1])

ax = fig.add_subplot(gs[0, 0])
ax.text(-0.10, 1.15, 'a', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.plot(ts, vels[:, 0], 'k-')
ax.set_xlim(ts[0], ts[-1])
ax.axvline(ts[all_inds[gap]], color = 'b')
ax.axvline(ts[all_inds[gap+1]], color = 'b')
ax.set_xlabel('time (minutes)')
ax.set_ylabel('velocity (m/s)')

ax = fig.add_subplot(gs[0, 1])
ax.text(-0.10, 1.15, 'b', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.imshow(diffs, cmap = 'coolwarm', extent = [ts[0], ts[-1], ts[-1], ts[0]])
ax.set_xlabel('time (minutes)')
ax.set_ylabel('time (minutes)')
ax.axvline(ts[all_inds[gap]], color = 'b')
ax.axvline(ts[all_inds[gap+1]], color = 'b')
ax.axhline(ts[all_inds[gap]], color = 'b')
ax.axhline(ts[all_inds[gap+1]], color = 'b')

ax = fig.add_subplot(gs[0, 2])
ax.text(-0.10, 1.15, 'c', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.plot(ts, lats[:, 14], 'k-')
ax.set_xlim(ts[0], ts[-1])
ax.set_xlabel('time (minutes)')
ax.set_ylabel('latent state (a.u.)')
ax.axvline(ts[all_inds[gap]], color = 'b')
ax.axvline(ts[all_inds[gap+1]], color = 'b')

plt.savefig('figures/quiescent.png', bbox_inches = 'tight')
plt.close()

data = {
    'ts': ts,
    'all_inds': all_inds,
    'gap': gap,
    'lats': lats,
    'vels': vels
}
pickle.dump(data, open('../figures/figure_data/primate/'+fname+'_quiescent_data.pickled', 'wb'))


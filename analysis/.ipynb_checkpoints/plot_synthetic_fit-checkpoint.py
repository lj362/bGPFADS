"""
Process data for synthetic plots and write to figure folder
use: python plot_synthetic_fit.py
"""

import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
from utils import basedir, detach
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

cm = 1/2.54
pi2 = 2*np.pi
plt.rcParams['font.size'] = 12

###### create figure ######
fig = plt.figure(figsize = (22*cm, 8*cm))

###### method comparison #####
gs = fig.add_gridspec(1, 2, left=0.00, right=1.00, bottom=0.0, top=1.00, 
                      wspace = 0.35, hspace = 0.15, width_ratios=[1, 1])

######### plot training log likelihood #######
ds_fit, LLs = pickle.load(open('data/LLs_nrep10_dmax10_mixed.pickled', 'rb'))

print(np.mean(LLs, axis = 0))
pickle.dump([ds_fit, LLs], open('../figures/figure_data/synthetic/train_LLs.pickled', 'wb'))

Nd = len(ds_fit) #number of dimensionalities
m, s = -np.mean(LLs, axis = 0), 1*np.std(LLs, axis = 0)

ax = fig.add_subplot(gs[0, 0])
ax.text(-0.10, 1.15, 'a', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.plot(ds_fit, m[:, 0], 'g-')
ax.fill_between(ds_fit, (m-s)[:, 0], (m+s)[:,0], color = 'g', alpha = 0.2)
ax.plot(ds_fit, m[:, 1], 'c-')
ax.fill_between(ds_fit, (m-s)[:, 1], (m+s)[:,1], color = 'c', alpha = 0.2)
ax.plot(ds_fit, m[:, 2], 'b-')
ax.fill_between(ds_fit, (m-s)[:, 2], (m+s)[:,2], color = 'b', alpha = 0.2)
ax.plot(ds_fit, np.ones(Nd)*m[0, 3],'k-')
ax.fill_between(ds_fit, np.ones(Nd)*(m-s)[0, 3], np.ones(Nd)*(m+s)[0,3], color = 'k', alpha = 0.2)

ax.set_ylabel('NLL')
ax.set_xlabel('latent dimensionality')
ax.legend(['FA', 'GPFA', 'BGPFA', 'BGPFA-ARD'], frameon = False)
ax.set_xlim(ds_fit[0], ds_fit[-1])


######### plot crossvalidated MSE #######
ds_fit, LLs, norm_MSEs, MSEs, LLs_train = pickle.load(open('data/cv_data_nrep10_dmax10_mixed.pickled', 'rb'))

print(np.round(np.mean(MSEs, axis = 0), 4))

RMSEs = MSEs ##actually plot MSEs not RMSEs in the end
m, s = np.mean(RMSEs, axis = 0), 1*np.std(RMSEs, axis = 0)
Nd = len(ds_fit) #number of dimensionalities

ax = fig.add_subplot(gs[0, 1])
ax.text(-0.10, 1.15, 'b', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')

ax.plot(ds_fit, m[:, 0], 'g-')
ax.fill_between(ds_fit, (m-s)[:, 0], (m+s)[:,0], color = 'g', alpha = 0.2)
ax.plot(ds_fit, m[:, 1], 'c-')
ax.fill_between(ds_fit, (m-s)[:, 1], (m+s)[:,1], color = 'c', alpha = 0.2)
ax.plot(ds_fit, m[:, 2], 'b-')
ax.fill_between(ds_fit, (m-s)[:, 2], (m+s)[:,2], color = 'b', alpha = 0.2)
ax.plot(ds_fit, np.ones(Nd)*m[0, 3],'k-')
ax.fill_between(ds_fit, np.ones(Nd)*(m-s)[0, 3], np.ones(Nd)*(m+s)[0,3], color = 'k', alpha = 0.2)

ax.set_ylabel('MSE')
ax.set_xlabel('latent dimensionality')
ax.set_xlim(ds_fit[0], ds_fit[-1])
pickle.dump([ds_fit, MSEs], open('../figures/figure_data/synthetic/cv_MSEs.pickled', 'wb'))
    
#### add inset ####
mod = torch.load('data/cv_data_nrep10_dmax10_mixed_bGPFA.pt')
_dim_scales = detach(mod.obs.dim_scale).flatten()
dim_scales = np.log(_dim_scales)
ms = np.sqrt(np.mean(detach(mod.lat_dist.nu)**2, axis = (0, -1)))

ax = fig.add_axes([0.7, 0.6, 0.15, 0.25])
ax.scatter(dim_scales, ms, color = 'k', s = 10)
pickle.dump([_dim_scales, ms], open('../figures/figure_data/synthetic/cv_ard_params.pickled', 'wb'))

plt.savefig('figures/LLs.png', bbox_inches = 'tight')
plt.close()


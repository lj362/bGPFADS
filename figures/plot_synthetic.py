import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
from scipy.stats import pearsonr
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

cm = 1/2.54
pi2 = 2*np.pi
plt.rcParams['font.size'] = 14

    
###### create figure ######
fig = plt.figure(figsize = (22*cm, 15*cm))

###### method comparison #####
gs = fig.add_gridspec(1, 2, left=0.00, right=1.00, bottom=0.585, top=1.00, 
                      wspace = 0.35, hspace = 0.15, width_ratios=[1, 1])
alpha = 0.1
######### plot training log likelihood #######
ab_cols = ['#F0E442', '#009E73', '#56B4E9']
ds_fit, LLs = pickle.load(open('figure_data/synthetic/train_LLs.pickled', 'rb'))
Nd = len(ds_fit) #number of dimensionalities

### max normalize across the FA/BFA methods separately ###
m, s = np.mean(LLs, axis = 0), 2*np.std(LLs, axis = 0)

j_LLs = np.load('figure_data/synthetic/jasmine_gpfa_LLs.npy')
m_j, s_j = np.mean(j_LLs, axis=0).reshape(-1, 1), 2*np.std(j_LLs, axis=0).reshape(-1, 1)
print(m_j)
m = np.concatenate([m, m_j], axis = 1)
s = np.concatenate([s, s_j], axis = 1) # jasmine gpfa
m_fa = max(np.amax(m[..., 0]), np.amax(m[..., 1]), np.amax(m[..., 4]))
m_bfa = max(np.amax(m[..., 2]), m[0, 3])
print(m_fa, m_bfa)
m[..., :2] = m[..., :2] #- m_fa
m[..., 4] = m[..., 4] #- m_fa
m[..., 2:4] = m[..., 2:4] #- m_bfa


ax = fig.add_subplot(gs[0, 0])
ax.text(-0.10, 1.15, 'a', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.plot(ds_fit, m[:, 0], ls = '-', marker = 'o', color = ab_cols[0])
ax.fill_between(ds_fit, (m-s)[:, 0], (m+s)[:,0], color = ab_cols[0], alpha = alpha)
ax.plot(ds_fit, m[:, 4], ls='-', marker = 'o', color = ab_cols[1])
ax.fill_between(ds_fit, (m-s)[:, 4], (m+s)[:,4], color = ab_cols[1], alpha = alpha)

ax.plot(ds_fit, m[:, 2], color=ab_cols[2], marker='o', ls='-')
ax.fill_between(ds_fit, (m-s)[:, 2], (m+s)[:,2], color = ab_cols[2], alpha = alpha)
ax.plot(ds_fit, np.ones(Nd)*m[0, 3],'k--')
ax.fill_between(ds_fit, np.ones(Nd)*(m-s)[0, 3], np.ones(Nd)*(m+s)[0,3], color = 'k', alpha = alpha)

ax.set_ylabel('LL/ELBO')
ax.set_xlabel('latent dimensionality')
ax.legend(['FA', 'GPFA', 'bGPFA (no ARD)', 'bGPFA (ARD)'], frameon = False, ncol = 2, loc = 'upper left', bbox_to_anchor = (-0.01, 1.175), columnspacing = 1.3, fontsize = 13, handletextpad=0.6)
ax.set_xlim(ds_fit[0]-0.1, ds_fit[-1]+0.1)
ax.set_yticks([-1.7, -1.6, -1.5, -1.4])

######### plot crossvalidated MSE #######
ds_fit, MSEs = pickle.load(open('figure_data/synthetic/cv_MSEs.pickled', 'rb'))
RMSEs = np.sqrt(MSEs)
RMSEs = MSEs #actually just use MSEs
m, s = np.mean(RMSEs, axis = 0), 2*np.std(RMSEs, axis = 0)

j_MSEs = np.load('figure_data/synthetic/jasmine_gpfa_MSEs.npy')
m_j, s_j = np.mean(j_MSEs, axis=0).reshape(-1, 1), 2*np.std(j_MSEs, axis=0).reshape(-1, 1)
print(m_j)
m = np.concatenate([m, m_j], axis = 1)
s = np.concatenate([s, s_j], axis = 1) # jasmine gpfa
print(np.round(m, 4))

ax = fig.add_subplot(gs[0, 1])
ax.text(-0.10, 1.15, 'b', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.plot(ds_fit, m[:, 0], ls='-', marker='o', color = ab_cols[0])
ax.fill_between(ds_fit, (m-s)[:, 0], (m+s)[:,0], color = ab_cols[0], alpha = alpha)
ax.plot(ds_fit, m[:, 4], color = ab_cols[1], marker = 'o', ls='-')
ax.fill_between(ds_fit, (m-s)[:, 4], (m+s)[:,4], color = ab_cols[1], alpha = alpha)

ax.plot(ds_fit, m[:, 2], color=ab_cols[2], marker='o', ls='-')
ax.fill_between(ds_fit, (m-s)[:, 2], (m+s)[:,2], color = ab_cols[2], alpha = alpha)
ax.plot(ds_fit, np.ones(Nd)*m[0, 3],'k--')
ax.fill_between(ds_fit, np.ones(Nd)*(m-s)[0, 3], np.ones(Nd)*(m+s)[0,3], color = 'k', alpha = alpha)

ax.set_ylabel('MSE')
ax.set_xlabel('latent dimensionality')
ax.set_xlim(ds_fit[0]-0.1, ds_fit[-1]+0.1)
ax.set_yticks(np.arange(1.2, 2.8, 0.4))

#### plot parameters as inset ####
_dim_scales, ms = pickle.load(open('figure_data/synthetic/cv_ard_params.pickled', 'rb'))
dim_scales = np.log(_dim_scales)
ax = fig.add_axes([0.72, 0.8, 0.18, 0.18])
ax.scatter(dim_scales, ms, color = 'k', s = 20, alpha = 0.6)
ax.set_xticks([-6, 0])
ax.set_yticks([0, 1])
ax.set_xlabel(r'$\log \, s_d$', labelpad = -10)
ax.set_ylabel(r'$||\nu_d||^2_2$', labelpad = -8)


#### plot example trajectories etc. ####
gs = fig.add_gridspec(1, 3, left=0.00, right=1.00, bottom=0.05, top=0.415, 
                      wspace = 0.45, hspace = 0.15, width_ratios=[1, 1, 1])
cols = ['#E69F00', '#0072B2', '#CC79A7']

### plot learned parameters ###
all_dim_scales = pickle.load(open('figure_data/synthetic/example_dim_scales.pickled', 'rb'))
all_ms = pickle.load(open('figure_data/synthetic/example_lat_ms.pickled', 'rb'))
all_ss = pickle.load(open('figure_data/synthetic/example_lat_ss.pickled', 'rb'))

#ax = fig.add_subplot(gs[0, 0], projection='3d')
ax = fig.add_subplot(gs[0, 0])
ax.text(-0.10, 1.15, 'c', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
for i in range(3):
    #kernel length scales
    _dim_scales = all_dim_scales[i]
    dim_scales = np.log(_dim_scales)
    #average over trials, timepoints
    ms = all_ms[i]
    ax.scatter(dim_scales, ms, c = cols[i], s = 30)
ax.set_xlabel(r'$\log \, s_d$')
ax.set_ylabel(r'$||\nu_d||^2_2$', labelpad = 5)
ax.legend(['Gaussian', 'Poisson', 'NegBinom'], frameon = False, loc = 'upper center', ncol = 1, bbox_to_anchor = [+0.3, 1.05], handletextpad=0.5)


### plot trajectories ###
xs, lats_gauss, lats_pois, lats_NB = pickle.load(open('figure_data/synthetic/example_latents.pickled', 'rb'))

ax = fig.add_subplot(gs[0, 1])
ax.text(-0.10, 1.15, 'd', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.plot(xs[:, 0], xs[:, 1], 'k-')
for i, lats in enumerate([lats_gauss, lats_pois, lats_NB]):
    plt.plot(lats[:, 0], lats[:, 1], c=cols[i], ls='-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('latent dim 1')
ax.set_ylabel('latent dim 2')
ax.legend([r'ground truth $\bf X$'], frameon = False, loc = 'upper center', bbox_to_anchor = [0.5, 1.15])


#### plot overdispersion parameters ####
r_true, r_inf, means = pickle.load(open('figure_data/synthetic/example_NB_overdispersion.pickled', 'rb'))
def plotfunc(r):
    return np.log(r)
    return r

lims = [f(np.concatenate([plotfunc(r_true), plotfunc(r_inf)])) for f in [np.amin, np.amax]]
lims = [lims[0]-0.1, lims[1]+0.1]

ax = fig.add_subplot(gs[0, 2])
ax.text(-0.15, 1.15, 'e', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.scatter(plotfunc(r_true), plotfunc(r_inf), c = 'k', s = 20)#20*means)
print(pearsonr(plotfunc(r_true), plotfunc(r_inf)))

trues = np.array(lims)
ax.plot(trues, trues, 'k-')
ax.set_xlabel(r'$\log \, \kappa_{true}$')
ax.set_ylabel(r'$\log \, \kappa_{inf}$')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks([0, 2, 4])
ax.set_yticks([0, 2, 4])

#### save figure ####
plt.savefig('synthetic_figure.pdf', bbox_inches = 'tight')
plt.close()

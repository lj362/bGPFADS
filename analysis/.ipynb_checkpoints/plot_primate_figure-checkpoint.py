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
from scipy.stats import binned_statistic
from utils import detach, basedir, not_in
from load_joey import load_data
device = mgp.utils.get_device()
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

cm = 1/2.54
pi2 = 2*np.pi
plt.rcParams['font.size'] = 12
device = mgp.utils.get_device("cuda")  # get_device("cpu")
np.random.seed(11110301)
torch.manual_seed(11110301)
binsize = '25'
d_fit = 25


def plot_decoding(ax, delays, all_vars, col = 'k', ls = '-', label = None):
        all_vars = np.mean(all_vars, axis = -1)
        m, s = np.mean(all_vars, axis = 0), np.std(all_vars, axis = 0) #mean, std over splits
        ax.plot(delays, m, col+ls, label = label)
        ax.fill_between(delays, m-2*s, m+2*s, color = col, alpha = 0.2)
        print(label, delays[np.argmax(m)], np.amax(m))
        
###### create figure ######
fig = plt.figure(figsize = (22*cm, 16*cm))

""
###### decoding #####
gs = fig.add_gridspec(1, 2, left=0.00, right=1.00, bottom=0.55, top=1.00, 
                      wspace = 0.25, hspace = 0.15, width_ratios=[1, 1])
ylim = [0, 0.7]
yticks = [0, 0.2, 0.4, 0.6]

### plot M1 and S1 decoding ###
ax = fig.add_subplot(gs[0, 0])
cols = ['k', 'b']
for i, region in enumerate(['M1', 'S1']):
    #fname = 'NegBinom_d30_'+region+'_b'+binsize+'_circ_subsamp'
    fname = 'NegBinom_d'+str(d_fit)+'_'+region+'_b'+binsize+'_circ'
    data = pickle.load(open('data/'+fname+'_move_kinematic_predictions.pickled', 'rb'))
    delays, all_vars, all_vars_con = data['delays'], data['bgpfa_sep'], data['convolved_sep']
    plot_decoding(ax, delays, all_vars, col = cols[i], ls = '-', label = region)
    plot_decoding(ax, delays, all_vars_con, col = cols[i], ls = '--')
    pickle.dump([delays, all_vars, all_vars_con], open('../figures/figure_data/primate/'+fname+'_kinematic_predictions.pickled', 'wb'))
ax.set_xlim(delays[0], delays[-1])
ax.set_xlabel('delay (ms)')
ax.set_ylabel('variance captured')
ax.set_ylim(ylim)
ax.set_yticks(yticks)
ax.legend(frameon = False)

''
### plot M1+S1 decoding ###
ax = fig.add_subplot(gs[0, 1])
cols = ['k', 'b']
for i, offset in enumerate([False, True]):
    
    if offset:
        #fname = 'NegBinom_d30_both_b'+binsize+'shift100_circ_subsamp'
        fname = 'NegBinom_d'+str(d_fit)+'_both_b'+binsize+'_shift100_circ'
        label = 'offset'
    else:
        #fname = 'NegBinom_d30_both_b'+binsize+'_circ_subsamp'
        fname = 'NegBinom_d'+str(d_fit)+'_both_b'+binsize+'_circ'
        label = 'M1+S1'
    
    data = pickle.load(open('data/'+fname+'_rep1_move_kinematic_predictions.pickled', 'rb'))
    delays, all_vars, all_vars_con = data['delays'], data['bgpfa_sep'], data['convolved_sep']
    plot_decoding(ax, delays, all_vars, col = cols[i], ls = '-', label = label)
    plot_decoding(ax, delays, all_vars_con, col = cols[i], ls = '--')
    
    pickle.dump([delays, all_vars, all_vars_con], open('../figures/figure_data/primate/'+fname+'_kinematic_predictions.pickled', 'wb'))

ax.set_xlim(delays[0], delays[-1])
ax.set_xlabel('delay (ms)')
ax.set_ylabel('variance captured')
ax.set_ylim(ylim)
ax.set_yticks(yticks)
ax.legend(frameon = False)
''
""


##### plot converged parameters #####

gs = fig.add_gridspec(1, 2, left=0.00, right=1.00, bottom=0.00, top=0.45, 
                      wspace = 0.25, hspace = 0.15, width_ratios=[1, 1])

names = ['M1_b'+binsize, 'S1_b'+binsize, 'both_b'+binsize, 'both_b'+binsize+'_shift100']
cols = ['k', 'r', 'b', 'g']
ax = fig.add_subplot(gs[0, 0], projection='3d')

for i, name in enumerate(names): 
    fname = 'NegBinom_d'+str(d_fit)+'_'+name+'_circ'
    mod = torch.load('data/'+fname+'.pt')
    #kernel length scales
    _dim_scales = detach(mod.obs.dim_scale).flatten()
    dim_scales = np.log(_dim_scales)
    ss = np.mean(detach(mod.lat_dist.scale), axis = (0, -1))
    ms = np.sqrt(np.mean(detach(mod.lat_dist.nu)**2, axis = (0, -1)))
    
    lambdas = _dim_scales**2 #compute participation ratio
    dimensionality = np.sum(lambdas)**2 / np.sum(lambdas**2) 
    
    args = np.argsort(dim_scales)
    print(names[i], dimensionality, '\n', dim_scales[args], '\n', ms[args], '\n', ss[args], '\n')
    ax.scatter(dim_scales, ms, ss, c = cols[i], s = 30)

    pickle.dump([_dim_scales, ms, ss], open('../figures/figure_data/primate/'+fname+'_model_params.pickled', 'wb'))
ax.set_xlabel(r'$\sigma_d$')
ax.set_ylabel(r'$||\mu||^2_2$')
ax.set_zlabel(r'$\langle s \rangle$')
ax.view_init(60, 120)


#### save figure ####
plt.savefig('figures/primate_fig.png', bbox_inches = 'tight')
plt.close()


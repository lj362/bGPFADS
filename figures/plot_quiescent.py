import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
from scipy.stats import pearsonr, linregress
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

cm = 1/2.54
pi2 = 2*np.pi
plt.rcParams['font.size'] = 14
binsize = 25
d_fit = 25
fname = 'NegBinom_d'+str(d_fit)+'_M1_b'+str(binsize)+'_circ'
    
###### create figure ######
fig = plt.figure(figsize = (26*4/3*cm, 5*4/3*cm))

gs = fig.add_gridspec(1, 3, left=0.00, right=1.00, bottom=0.0, top=1.00, 
                      wspace = 0.15, hspace = 0.35, width_ratios = [1, 0.7, 1])

        
### plot reaction time figure
data = pickle.load(open('figure_data/primate/'+fname+'_quiescent_data.pickled', 'rb'))    
ts = data['ts']
gap = data['gap']
lats = data['lats']
all_inds = data['all_inds']
vels = data['vels']
vs = np.sqrt(np.sum(vels**2, axis = -1))

print('downsampling')
nsamp = 8
lats_sub = np.array([np.mean(lats[i*nsamp:(i+1)*nsamp, :], axis = 0) for i in range(int(np.floor(lats.shape[0]/nsamp)))])
print(lats_sub.shape)
print('computing differences')
diffs = lats_sub[None, ...] - lats_sub[:, None, :] # T x T x d
diffs = np.sqrt(np.sum(diffs**2, axis = -1))

tg1, tg2 = ts[all_inds[gap]], ts[all_inds[gap+1]]
ax = fig.add_subplot(gs[0, 0])
ax.text(-0.10, 1.15, 'a', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.plot(ts, vs, 'k-')
ax.set_xlim(ts[0], ts[-1])
ymin, ymax = 0, 0.85
ax.set_ylim(ymin, ymax)
ax.axvline(tg1, color = 'b')
ax.axvline(tg2, color = 'b')
ax.fill_between([tg1, tg2], [ymin, ymin], [ymax, ymax], color = 'b', alpha = 0.05)
ax.set_xlabel('time (minutes)')
ax.set_ylabel('speed (m/s)')

ax = fig.add_subplot(gs[0, 1])
ax.text(-0.15, 1.15, 'b', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.imshow(-diffs, cmap = 'coolwarm', extent = [ts[0], ts[-1], ts[-1], ts[0]])
ax.set_xlabel('time (minutes)')
ax.set_ylabel('time (minutes)')
ax.axvline(tg1, color = 'b')
ax.axvline(tg2, color = 'b')
ax.axhline(tg1, color = 'b')
ax.axhline(tg2, color = 'b')

ax = fig.add_subplot(gs[0, 2])
ax.text(-0.05, 1.15, 'c', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.plot(ts, -lats[:, 14], 'k-')
ymin, ymax = -13, 5
ax.set_xlim(ts[0], ts[-1])
ax.set_ylim(ymin, ymax)
ax.set_xlabel('time (minutes)')
ax.set_ylabel('latent state (a.u.)')
ax.set_yticks([])
ax.axvline(tg1, color = 'b')
ax.axvline(tg2, color = 'b')
ax.fill_between([tg1, tg2], [ymin, ymin], [ymax, ymax], color = 'b', alpha = 0.05)

    
#### save figure ####
plt.savefig('supp_quiescent.pdf', bbox_inches = 'tight')
plt.close()
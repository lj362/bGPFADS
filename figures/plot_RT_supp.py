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
loadstr = '_thresh'
    
###### create figure ######
fig = plt.figure(figsize = (26*2.5/3*cm, 15*cm))

gs = fig.add_gridspec(1, 2, left=0.00, right=1.00, bottom=0.585, top=1.00, 
                      wspace = 0.35, hspace = 0.35)

        
### plot reaction time figure
data = pickle.load(open('figure_data/primate/'+fname+'_RTs_dists'+loadstr+'.pickled', 'rb'))    
lat_diffs_sub, RTs_sub, RTs = data
corr = pearsonr(lat_diffs_sub, RTs_sub)[0]
print('corr:', corr)

### plot RT histogram ###
RTmin, RTmax = 125, 425
RT_hist = pickle.load(open('figure_data/primate/'+fname+'_RT_hist'+loadstr+'.pickled', 'rb'))
bins = np.arange(0, 1000, binsize)
print('min RT:', np.amin(RT_hist))

ax = fig.add_subplot(gs[0, 0])
ax.text(-0.10, 1.15, 'a', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
print(np.amin(RT_hist), np.amax(RT_hist))
ax.hist(RT_hist, color = 'k', bins = bins)
ax.axvline(RTmin, color = 'b', lw=2)
ax.axvline(RTmax, color = 'b', lw=2)
ax.set_xlabel('reaction time (ms)')
ax.set_ylabel('frequency')
print('RTs:', len(RTs), len(RTs[(RTs >= RTmin) & (RTs <= RTmax)]))

### plot null distribution from generative model ###
all_rs = pickle.load(open('figure_data/primate/'+fname+'_RTs_synthetic'+loadstr+'.pickled', 'rb'))
print('minimax r:', np.amin(all_rs), np.amax(all_rs))
print('mean std r:', np.mean(all_rs), np.std(all_rs))
bins = np.linspace(-0.2, 0.2, 100)

ax = fig.add_subplot(gs[0, 1])
ax.text(-0.10, 1.15, 'b', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.hist(all_rs, color = 'k', bins = bins)
ax.axvline(np.mean(all_rs), color = 'b', ls = '--', lw=2)
ax.axvline(corr, color = 'b', lw=2)
ax.legend(['synthetic\nmean', 'data'], frameon = False, bbox_to_anchor=(0.865,1.01), loc = 'upper right', fontsize = 13)
ax.set_xlim(-0.15, 0.50)
ax.set_xlabel('correlation with RT')
ax.set_ylabel('frequency')


#### plot trial length histogram and correlation with latent state####
gs = fig.add_gridspec(1, 2, left=0.00, right=1.00, bottom=0.0, top=0.415, 
                      wspace = 0.35, hspace = 0.35)
data = pickle.load(open('figure_data/primate/'+fname+'_RTs_lats'+loadstr+'.pickled', 'rb'))    
lats, all_RTs, Ts = data
RT_args_tot = np.where( (all_RTs <= RTmax) & (all_RTs >= RTmin) )[0]
lats_sub, all_RTs_sub, Ts_sub = [dat[RT_args_tot] for dat in data]
lats_sub = +lats_sub

### plot histogram of reach durations ###
ax = fig.add_subplot(gs[0, 0])
ax.text(-0.10, 1.15, 'c', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.hist(Ts_sub, color = 'k', bins = np.arange(800, 4000, 100))
ax.set_xlabel('reach duration (ms)')
ax.set_ylabel('frequency')
print('min max durations:', np.amin(Ts_sub), np.amax(Ts_sub))
ax.set_xticks(np.arange(1000, 4001, 1000))

### plot latent state correlation ###
print(pearsonr(lats_sub, all_RTs_sub))
ax = fig.add_subplot(gs[0, 1])
ax.text(-0.10, 1.15, 'd', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
plt.scatter(lats_sub, all_RTs_sub, c = 'k', marker = 'o', alpha = 0.5, s = 5)
ax.set_xlabel('latent state (a.u.)')
ax.set_ylabel('reaction time (ms)')
ax.set_xticks([])

s, i, _, _, _ = linregress(lats_sub, all_RTs_sub)
print(s, i)
xs = np.array([np.amin(lats_sub), np.amax(lats_sub)])
ax.plot(xs, s*xs+i, 'k-')
ax.set_yticks([200, 300, 400])
    
#### save figure ####
plt.savefig('supp_RT_figure.pdf', bbox_inches = 'tight')
plt.close()
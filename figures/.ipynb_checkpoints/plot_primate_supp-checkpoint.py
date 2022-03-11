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
binsize = 25
d_fit = 25
fname = 'NegBinom_d'+str(d_fit)+'_M1_b'+str(binsize)+'_circ'

loadstr = '_thresh'
    
###### create figure ######
fig = plt.figure(figsize = (26*cm, 12*cm))


###### heatmaps #####
gs = fig.add_gridspec(2, 2, left=0.00, right=0.45, bottom=0.0, top=1.00, 
                      wspace = 0.35, hspace = 0.35)

all_labs = [['Y_init', 'Y_move'], ['FA_init', 'FA_move']]
labs = [r'$\mathbf{Y}$', 'FA']
naxs = [[(0,0), (1,0)], [(0,1), (1,1)]]
panels = [['a', 'b'], ['c', 'd']]
for i in range(2):
    all_sims = [pickle.load(open('figure_data/primate/'+fname+'_similarity_mat_'+lab+loadstr+'.pickled', 'rb')) for lab in all_labs[i]]
    sims1, sims2 = all_sims
    all_sims = np.concatenate(all_sims)
    vmin, vmax = np.nanquantile(all_sims, 0.05)  , np.nanquantile(all_sims, 0.95) 
    
    ax1 = fig.add_subplot(gs[naxs[i][0][0], naxs[i][0][1]])
    ax2 = fig.add_subplot(gs[naxs[i][1][0], naxs[i][1][1]])   
    

    ax1.imshow(sims1, cmap = 'coolwarm', vmin = vmin, vmax = vmax)
    im = ax2.imshow(sims2, cmap = 'coolwarm', vmin = vmin, vmax = vmax)
    ax1.text(-0.10, 1.15, panels[i][0], transform=ax1.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
    ax2.text(-0.10, 1.15, panels[i][1], transform=ax2.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
    ax1.set_title('target onset ('+labs[i]+')', fontsize = 15)
    ax2.set_title('pre-movement ('+labs[i]+')', fontsize = 15)
    
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('reach direction')
        ax.set_ylabel('reach direction')

gs = fig.add_gridspec(1, 1, left=0.4725, right=0.4875, bottom=0.05, top=0.95)
ax = fig.add_subplot(gs[0, 0])

cbar = plt.colorbar(im, orientation = 'vertical', cax = ax, fraction = 0.05, shrink = 1, ticks=[vmin, vmax])
cbar.ax.set_yticklabels(['min sim.', 'max sim.'], rotation = 270, va = 'center', ha = 'left')

        

### curve plot (panel e)###
gs = fig.add_gridspec(2, 2, left=0.615, right=0.90, bottom=0.0, top=1.00, wspace = 0.25, hspace = 0.35)

labs = [['bGPFA_init', 'Y_init', 'FA_init'], ['bGPFA_move', 'Y_move', 'FA_move']]
cols = ['k', '#D55E00', '#009E73', '#0072B2']
titles = ['target onset', 'pre-movement']
ylims = [-0.5, 0.8]
yticks = [-0.5, 0.0, 0.5]
    
for i in range(2):
    ax = fig.add_subplot(gs[0, i])
    ax.set_title(titles[i], fontsize = 15)
    
    for j, lab in enumerate(labs[i]):
        xs, means, sems = pickle.load(open('figure_data/primate/'+fname+'_avg_similarity_'+lab+loadstr+'.pickled', 'rb'))
        ax.plot(xs, -means, color=cols[j], ls='-')
    ax.set_ylim(ylims)
    ax.set_xlim(-np.pi, np.pi)
    
    if i == 0:
        ax.set_xticks([])
        ax.legend(['bGPFA', 'Y', 'FA'], frameon = False, labelspacing = 0.4, fontsize='small', loc = 'upper center', handlelength=1.5)
        #ax.set_yticks([-0.6, -0.3, 0.0, 0.3])
        ax.set_yticks(yticks)
        ax.set_ylabel('similarity (z-score)')
        ax.text(-0.30, 1.15, 'e', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
    else:
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax.set_xlabel(r'$\Delta \theta$')

        
### plot modulation over time ####

all_mods = []
ts = []
RT_rs = []

shifts = range(-8, 10)
shifts = list(shifts)
for Lmax_shift in shifts:
    mod_labs, modulation = pickle.load(open('figure_data/primate/'+fname+'_avg_similarity_modulation_'+str(Lmax_shift)+loadstr+'.pickled', 'rb'))
    all_mods.append(modulation)
    ts.append(Lmax_shift*binsize)
    
vels, move_bins = pickle.load(open('figure_data/primate/'+fname+'_vels'+loadstr+'.pickled', 'rb'))
speeds = [np.sqrt(np.sum(vel**2, axis = -1)) for vel in vels]
print(len(vels))
all_speeds = np.concatenate(vels, axis = 0)
print('vels:', all_speeds.shape)
def transform(X):
    return (X - np.mean(all_speeds))/np.std(all_speeds)

speeds = [transform(speed) for speed in speeds]
speed_trajs = np.zeros((len(speeds), len(shifts)))*np.nan
imove = shifts.index(0)
print(imove+shifts[0], imove+shifts[-1]+1, len(shifts))
for ireach, speed in enumerate(speeds):
    newmove = move_bins[ireach]
    
    indmin = max(-newmove, shifts[0])
    indmax = min(len(speed)-newmove, shifts[-1]+1)
    
    speed_trajs[ireach, (indmin+imove):imove] = speed[(newmove+indmin):newmove]
    speed_trajs[ireach, imove:(imove+indmax)] = speed[newmove:(newmove+indmax)]

print(speed_trajs.shape)
print(np.nanmean(speed_trajs, axis = 0))
meanspeeds = np.nanmean(speed_trajs, axis = 0)
    
all_mods = np.array(all_mods)
print(RT_rs)

print(ts)
print(np.round(all_mods, 3).T)

ax = fig.add_subplot(gs[1, :])

ax.text(-0.10, 1.15, 'f', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
for i in range(3):
    ax.plot(ts, all_mods[:, i], color=cols[i], ls='--')
    ax.plot(ts, all_mods[:, i+4], color=cols[i],ls='-')
ax.plot(ts, meanspeeds, color = cols[3], ls = '-', label = 'hand speed')
ax.set_xlabel('time from movement onset (ms)')
ax.set_ylabel('modulation')
ax.set_yticks(np.arange(0, 1.6, 0.5))
ax.set_xticks(np.arange(-200, 200, 100))
ax.set_xlim(ts[0], ts[-1])

ax.set_xlim(ts[0], -ts[0])
ax.set_xticks(np.arange(-200, 250, 100))
ax.set_yticks([0, 1, 2])
ax.set_ylabel('modulation/speed')
ax.legend(frameon = False, fontsize='small', loc = 'upper left', handlelength=1.5)
                 
                 
#### save figure ####
plt.savefig('supp_primate_figure.pdf', bbox_inches = 'tight')
plt.close()
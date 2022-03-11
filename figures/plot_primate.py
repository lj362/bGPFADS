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

loadstr = '_thresh'

def plot_decoding(ax, delays, all_vars, col = 'k', ls = '-', label = None):
        all_vars = np.mean(all_vars, axis = -1)
        m, s = np.mean(all_vars, axis = 0), np.std(all_vars, axis = 0) #mean, std over splits
        ax.plot(delays, m, col+ls, label = label)
        ax.fill_between(delays, m-2*s, m+2*s, color = col, alpha = 0.2)
        print(label, delays[np.argmax(m)], np.amax(m))
        
###### create figure ######
fig = plt.figure(figsize = (26*4/3*cm, 17*cm))

binsize = 25
d_fit = 25

gs = fig.add_gridspec(1, 4, left=0.00, right=1.00, bottom=0.585, top=1.00, 
                      wspace = 0.35, hspace = 0.15)


##### plot schematics ######
ax = fig.add_subplot(gs[0, 0])
ax.text(-0.10, 1.15, 'a', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')

### plot 8x8 grid ####

rad = 0.2
for x in range(8):
    for y in range(8):
        circle = plt.Circle((x, y), rad, color=[0.5, 0.5, 0.5])
        ax.add_patch(circle)
        
coords = [np.array([1, 1]), np.array([5, 4]), np.array([2, 6])]
cols = [np.array([0.2, 0.2, 0.8]), np.array([0.2, 0.5, 0.5]), np.array([0.2, 0.8, 0.2])]

for i in range(3):
    c = plt.Circle(coords[i], rad, color=cols[i])
    ax.add_patch(c)
for i in range(2):
    dx, dy = coords[i+1][0]-coords[i][0], coords[i+1][1]-coords[i][1]
    plt.arrow(coords[i][0], coords[i][1], dx, dy, color = 0.5*(cols[i+1]+cols[i]),
              head_width = 0.4, head_length = 0.6, length_includes_head = True, lw = 2.5)
    
### plot theta 1 ####
ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[0][1]], ls = '--', color = 0.5*(cols[0]+cols[1]), lw = 2.5)
ax.text(coords[0][0]+2.5, coords[0][1]+1, r'$\theta_1$',fontsize=20, fontweight='bold', va='top', ha='right', color = 0.5*(cols[0]+cols[1]))
    
### plot theta 2 ####
ax.plot([coords[1][0], 7], [coords[1][1], coords[1][1]], ls = '--', color = 0.5*(cols[1]+cols[2]), lw = 2.5)
ax.text(coords[1][0]+1, coords[1][1]+1, r'$\theta_2$',fontsize=20, fontweight='bold', va='top', ha='right', color = 0.5*(cols[1]+cols[2]))
    

ax.axis('off')
ax.plot([], [])
ax.set_title('task structure')


###### decoding #####

ylim = [0.1, 0.7]
yticks = [0.1, 0.3, 0.5, 0.7]

### plot M1 and S1 decoding ###
ax = fig.add_subplot(gs[0, 2])
ax.text(-0.10, 1.15, 'c', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
cols = ['k', 'b']
for i, region in enumerate(['M1', 'S1']):
    fname = 'NegBinom_d'+str(d_fit)+'_'+region+'_b'+str(binsize)+'_circ'
    delays, all_vars, all_vars_raw = pickle.load(open('figure_data/primate/'+fname+'_kinematic_predictions.pickled', 'rb'))
    plot_decoding(ax, delays, all_vars, col = cols[i], ls = '-', label = region)
    plot_decoding(ax, delays, all_vars_raw, col = cols[i], ls = '--')
ax.set_xlim(delays[0], delays[-1])
ax.set_xlabel('delay (ms)')
ax.set_ylabel('variance captured')
ax.set_ylim(ylim)
ax.set_yticks(yticks)
ax.legend(frameon = False, loc = 'upper left')


### plot M1+S1 decoding ###
ax = fig.add_subplot(gs[0, 3])
ax.text(-0.10, 1.15, 'd', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
cols = ['c', 'g']
for i, offset in enumerate([False, True]):
    
    if offset:
        fname = 'NegBinom_d'+str(d_fit)+'_both_b'+str(binsize)+'_shift100_circ'
        label = '100ms shift'
    else:
        fname = 'NegBinom_d'+str(d_fit)+'_both_b'+str(binsize)+'_circ'
        label = 'M1 & S1'
    
    delays, all_vars, all_vars_raw = pickle.load(open('figure_data/primate/'+fname+'_kinematic_predictions.pickled', 'rb'))
    plot_decoding(ax, delays, all_vars, col = cols[i], ls = '-', label = label)
    plot_decoding(ax, delays, all_vars_raw, col = cols[i], ls = '--')
ax.set_xlim(delays[0], delays[-1])
ax.set_xlabel('delay (ms)')
ax.set_ylabel('variance captured')
ax.set_ylim(ylim)
ax.set_yticks(yticks)
ax.legend(frameon = False)


##### plot converged parameters #####

cols = ['k', 'b', 'c', 'g']
names = ['M1_b'+str(binsize), 'S1_b'+str(binsize), 'both_b'+str(binsize), 'both_b'+str(binsize)+'_shift100']
ax = fig.add_subplot(gs[0, 1])
ax.text(-0.10, 1.15, 'b', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
for i, name in enumerate(names):
    _dim_scales, ms, ss = pickle.load(open('figure_data/primate/NegBinom_d'+str(d_fit)+'_'+name+'_circ_model_params.pickled', 'rb'))
    #kernel length scales
    dim_scales = np.log(_dim_scales)
    #average over trials, timepoints
    args = np.argsort(-_dim_scales)
    print(names[i], '\n', np.round(dim_scales[args], 3), '\n', np.round(ms[args], 3), '\n')
    ax.scatter(dim_scales, ms, c = cols[i], s = 30, alpha = 0.5, marker = 'x')
    
ax.set_xlabel(r'$\log \, s_d$')
ax.set_ylabel(r'$||\nu_d||^2_2$')
ax.legend(['M1', 'S1', 'M1 & S1', '100ms shift'], frameon = False, handletextpad=0.5, markerscale = 1.5)
ax.set_yticks(np.arange(0, 0.9, 0.2))



##### plot latent trajectories ####

gs = fig.add_gridspec(1, 7, left=0.00, right=1.00, bottom=0.00, top=0.415, 
                      hspace = 0.15, width_ratios = [1, 0.25, 1, 0.04, 1.1, 0.4, 1], wspace = 0)

region = 'M1'
binsize = 25
d_fit = 25

fname = 'NegBinom_d'+str(d_fit)+'_'+region+'_b'+str(binsize)+'_circ'

lats, angs, Lmaxs, RTs = pickle.load(open('figure_data/primate/'+fname+'_latents_angs'+loadstr+'.pickled', 'rb'))

#right
args1 = np.argsort( (angs + 0*np.pi)**2 )[:40]
lats1 = [lats[a] for a in args1]
Lmaxs1 = [Lmaxs[a] for a in args1]
ireaches = [18, 17, 26, 23, 14] #example reaches right
print(angs[args1])

#left
args2 = np.argsort( (angs - 1*np.pi)**2 )[:40]
lats2 = [lats[a] for a in args2]
Lmaxs2 = [Lmaxs[a] for a in args2]
ireaches2 = [1, 2, 4, 12, 17] #example reaches left
print(angs[args2])

means = [np.mean(np.array([newlat[i][Lmaxs[i], :] for i in range(40)]), axis = 0) for newlat in [lats1, lats2]]
print(means[0][:2], means[1][:2])

i1, i2 = 0, 1
ax = fig.add_subplot(gs[0, 0])
ax.text(-0.10, 1.15, 'e', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
#for ireach in range(15, 18):
for dirnum, trajs in enumerate([lats, lats2][:2]):
    for reachnum, ireach in enumerate([ireaches, ireaches2][dirnum]):
        newlats = trajs[ireach]
        L = newlats.shape[0]-1
        Lmax = [Lmaxs, Lmaxs2][dirnum][ireach]+3 #plot until movement onset

        Lplot = Lmax+0
        print(ireach, Lmax, L, newlats[Lmax, i1], newlats[Lmax,i2])#, np.sqrt(np.sum((newlats[Lmax, :]-means[dirnum])[:2]**2)))
        for l in range(Lplot):
            if dirnum == 0:
                col = 0.7*np.ones(3) * reachnum / len(ireaches)
                #label = r'$0^\circ$' if reachnum == len(ireaches)-1 and l == 0 else None
                label = 'right' if reachnum == len(ireaches)-1 and l == 0 else None
            else:
                col = np.array([0.50,0.00,0.08]) * (1.5*reachnum / len(ireaches2) + 0.5)
                #label = r'$180^\circ$' if reachnum == len(ireaches2)-1 and l == 0 else None
                label = 'left' if reachnum == len(ireaches2)-1 and l == 0 else None
            if l >= Lmax: col = 'g'
                
            ax.plot(newlats[l:l+2, i1], newlats[l:l+2, i2], ls = '-', color = col, alpha = 1, label = label)
        ax.scatter([newlats[Lmax, i1]], [newlats[Lmax,i2]], color = col, s = 30, marker = 'o', alpha = 1)
ax.legend(frameon = False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('latent dim 1')
ax.set_ylabel('latent dim 2')

##### plot latent similarities at init and move ####
labs = ['bGPFA_init', 'bGPFA_move']

init_lat_sims, move_lat_sims = [pickle.load(open('figure_data/primate/'+fname+'_similarity_mat_'+lab+loadstr+'.pickled', 'rb')) for lab in labs]

all_lat_sims = np.concatenate([init_lat_sims, move_lat_sims])
vmin, vmax = np.nanquantile(all_lat_sims, 0.01)  , np.nanquantile(all_lat_sims, 0.99)    
    
ax1, ax2 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 4])
ax1.imshow(init_lat_sims, cmap = 'coolwarm', vmin = vmin, vmax = vmax)
im = ax2.imshow(move_lat_sims, cmap = 'coolwarm', vmin = vmin, vmax = vmax)

cbar = plt.colorbar(im, orientation = 'vertical', ax = ax2, fraction = 0.05, shrink = 1, ticks=[vmin, vmax])
cbar.ax.set_yticklabels(['min sim.', 'max sim.'], rotation = 270, va = 'center', ha = 'left')

ax1.text(-0.10, 1.15, 'f', transform=ax1.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax1.set_title('target onset')
ax2.set_title('pre-movement')
for ax in [ax1, ax2]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('reach direction')
ax1.set_ylabel('reach direction')

                        

#### plot RT correlation ####
data = pickle.load(open('figure_data/primate/'+fname+'_RTs_dists'+loadstr+'.pickled', 'rb'))    
lat_diffs_sub, RTs_sub, RTs = data

ax = fig.add_subplot(gs[0, 6])
ax.text(-0.10, 1.15, 'g', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
plt.scatter(lat_diffs_sub, RTs_sub, c = 'k', marker = 'o', alpha = 0.5, s = 5)
ax.set_xlabel('distance to prep (a.u.)')
ax.set_ylabel('reaction time (ms)', labelpad = -10)

s, i, _, _, _ = linregress(lat_diffs_sub, RTs_sub)
print(s, i)
print(pearsonr(lat_diffs_sub, RTs_sub))
xs = np.array([np.amin(lat_diffs_sub), np.amax(lat_diffs_sub)])
ax.plot(xs, s*xs+i, 'k-')
#ax.set_yticks([200, 600])
ax.set_yticks([125, 425])
ax.set_xticks([])
        
#### save figure ####
plt.savefig('primate_figure.pdf', bbox_inches = 'tight')
plt.close()




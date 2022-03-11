import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import sys
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
np.random.seed(8432603)
np.random.seed(8522603)

cm = 1/2.54
pi2 = 2*np.pi
plt.rcParams['font.size'] = 16

T = 150
ts = np.arange(T)
dts_sq = (ts[:, None] - ts[None, :])**2
ell = 17
K = np.exp(-dts_sq/(2*ell**2))
L = np.linalg.cholesky(K + 1e-6*np.eye(T))

xs = np.linspace(-2, 2, T)
Kx = xs[:, None] * xs[None, :]
Lx = np.linalg.cholesky(Kx + 1e-6*np.eye(T))

###### create figure ######
fig = plt.figure(figsize = (30*cm, 10*cm))


###### plot priors #####
gs = fig.add_gridspec(2, 1, left=0.00, right=0.22, bottom=0.00, top=1.00, 
                      wspace = 0.15, hspace = 0.35)

### plot RBF prior ###
nex = 4
fs = L @ np.random.normal(0, 1, (T, nex))

ax = fig.add_subplot(gs[0, 0])
for i in range(nex):
    ax.plot(ts, fs[:, i], 'k--')
ax.fill_between([ts[0], ts[-1]], [-2, -2], [2, 2], color = 'k', alpha = 0.10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(ts[0], ts[-1])
ax.set_ylim(-3, 3)
ax.set_xlabel('time')
ax.set_ylabel( 'latent state')
ax.set_title('prior')

### plot linear prior ###
nex = 4
fxs = Lx @ np.random.normal(0, 1, (T, nex))

ax = fig.add_subplot(gs[1, 0])
for i in range(nex):
    ax.plot(xs, fxs[:, i], 'k--')
ax.fill_between(xs, -2*xs, 2*xs, color = 'k', alpha = 0.10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(xs[0], xs[-1])
ax.set_xlabel( 'latent state')
ax.set_ylabel('activity (neuron i)')


#### plot noise process
gs = fig.add_gridspec(1, 1, left=0.30, right=0.36, bottom=0.00, top=1.00, 
                      wspace = 0.15, hspace = 0.35)
ax = fig.add_subplot(gs[0, 0])
arr_image = plt.imread('dice.png', format='png')
ax.imshow(arr_image)
ax.axis('off')


#### plot data ####
gs = fig.add_gridspec(1, 1, left=0.44, right=0.66, bottom=0.275, top=0.725, 
                      wspace = 0.15, hspace = 0.35)
N = 50
C = np.random.normal(0, 1, (N, nex))*0.2
F = C @ fs.T
#Y = F + np.random.normal(0, 0.5, F.shape)
Y = np.random.poisson(np.exp(F+0.1))
print(np.amax(Y), np.quantile(Y, 0.8))

ax = fig.add_subplot(gs[0, 0])
ax.imshow(Y, aspect = 'auto', cmap = 'Greys', interpolation = 'none', vmin = 0, vmax = np.amax(Y)-3)
ax.spines.left = False
ax.spines.bottom = False
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('time')
ax.set_ylabel('neuron')


gs = fig.add_gridspec(2, 1, left=0.78, right=1.0, bottom=0, top=1, 
                      wspace = 0.15, hspace = 0.35)

### plot latent ###
nex_post = 2
fs_post = L @ np.random.normal(0, 1, (T, nex_post))
ax1 = fig.add_subplot(gs[0, 0])
for i in range(nex_post):
    ax1.plot(ts, fs_post[:, i], 'b-')
    ax1.fill_between(ts, fs_post[:, i]-0.3, fs_post[:, i]+0.3, color = 'b', alpha = 0.10)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlim(ts[0], ts[-1])
ax1.set_ylim(-3, 3)
ax1.set_xlabel('time')
ax1.set_ylabel( 'latent state')
ax1.set_title('posterior')

### plot activity ###
np.random.seed(8392603)
fxs_post = Lx @ np.random.normal(0, 1, (T, nex_post))

ax2 = fig.add_subplot(gs[1, 0])
for i in range(nex_post):
    ax2.plot(xs, fxs_post[:, i], 'b-', label = ('signal' if i == 0 else None))
    ax2.fill_between(xs, fxs_post[:, i]*0.7, fxs_post[:, i]*1.3, color = 'b', alpha = 0.10)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlim(xs[0], xs[-1])
ax2.set_xlabel( 'latent state')
ax2.set_ylabel('activity (neuron i)')


def get_con(ax, xyA, xyB, con_lw = 2):
    con = ConnectionPatch(xyA=xyA, xyB=xyB, arrowstyle="->",
                          coordsA='figure fraction', coordsB='figure fraction',
                          axesA=ax, axesB=ax, lw = con_lw)
    ax.add_artist(con)


x, dy = 0.75, 0.20

get_con(ax, (0.225, 0.5+dy), (0.295, 0.525)) #latents to dice
get_con(ax, (0.225, 0.5-dy), (0.295, 0.475)) #data to dice

get_con(ax, (0.355, 0.5), (0.415, 0.5)) #dice to data

get_con(ax, (0.665, 0.525), (x, 0.5+dy)) #data to latents
get_con(ax, (0.665, 0.475), (x, 0.5-dy)) #data to tuning


### plot latent ###
for i in range(nex_post):
    ax1.plot(ts, np.zeros(T), 'k-')
    ax1.fill_between(ts, np.zeros(T)-2, np.zeros(T)+2, color = 'k', alpha = 0.10)

### plot activity ###
for i in range(nex_post):
    ax2.plot(xs, np.zeros(T), 'k-', label = ('noise' if i == 0 else None))
ax2.legend(frameon = False, ncol = 1, loc = 'upper left', bbox_to_anchor = (-0.01, 1.15))


#### save figure ####
plt.savefig('schematic_figure.pdf', bbox_inches = 'tight')
plt.close()
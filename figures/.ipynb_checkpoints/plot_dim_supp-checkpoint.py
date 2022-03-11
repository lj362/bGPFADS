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
fname = 'NegBinom_d20_M1_b'+str(binsize)+'_circ'

if len(sys.argv) > 1:
    basedir = sys.argv[1] #user specified data directory
else:
    basedir = '/scratches/sagarmatha/gplvm/'
    
###### create figure ######
fig = plt.figure(figsize = (9*cm, 6*cm))

gs = fig.add_gridspec(1, 1, left=0.00, right=1.00, bottom=0.0, top=1.00, 
                      wspace = 0.35, hspace = 0.35)

        
### load data
shifts, prs = pickle.load(open('figure_data/primate/PR_by_offset.pickled', 'rb'))  


print('min shifts:', shifts[np.argmin(prs)])

### plot dimensionalities ###
ax = fig.add_subplot(gs[0, 0])
ax.text(-0.10, 1.15, 'a', transform=ax.transAxes,fontsize=25, fontweight='bold', va='top', ha='right')
ax.scatter(shifts, prs, marker = 'o', c = 'k', s = 30)
ax.set_xlabel('M1 spike time shift (ms)')
ax.set_ylabel('participation ratio')

    
#### save figure ####
plt.savefig('supp_dim_figure.pdf', bbox_inches = 'tight')
plt.close()
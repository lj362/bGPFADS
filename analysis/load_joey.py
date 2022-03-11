import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from utils import basedir, detach
np.random.seed(11101401)

def raster(labs, dat, bins, thresh, dt, shift, shiftsize, binshift):
    #### bin the spikes ####
    Y = []
    for lab in labs:
        allspikes = dat['spikes'] if lab == 'M1' else dat['S1_spikes']
        for unum, spikes in allspikes.items():
            #print(unum, spikes.size/dt)
            if spikes.size >= (thresh*dt): #require >= thresh Hz
                if shift and lab == 'M1':
                    spikes += (shiftsize*1e-3) 
                peth = np.histogram(spikes, bins = bins)[0] #compute peth
                
                if shift and shiftsize > 0:
                    peth = peth[binshift:]
                elif shift and shiftsize < 0:
                    peth = peth[:-binshift]
                
                Y.append(peth)
                
    Y = np.array(Y)[None, ...] #n_samples x n x T
    return Y

##### load data ######
def load_data(name = 'indy_20160426_01', region = 'M1', binsize = 50, subsamp = False, behavior = True,
              shift = False, thresh = 2, shiftsize = 100, convolve = False, lens = 50):

    dat = pickle.load(open('data/'+name+'.pickled', 'rb'))
    ts = dat['time']
    dt = ts[-1] - ts[0]
    bins = np.arange(ts[0], ts[-1], binsize*1e-3) 
    binshift = int(np.ceil(np.abs(shiftsize/binsize))) # #bins to discard
    labs = ['M1', 'S1'] if region == 'both' else [region]
    
    if convolve:
        Y_nc = raster(labs, dat, bins, thresh, dt, shift, shiftsize, binshift)
        """sample at (binsize/10)ms, convolve with a Gaussian filter of width 50ms (ref autolfads), subsamp to binsize"""
        newbins = np.arange(ts[0], ts[-1], binsize*1e-4)
        Yraw = raster(labs, dat, newbins, thresh, dt, shift, shiftsize, int(10*binshift))[0, ...] # N x T
        print('width:', lens/binsize*10)
        Yraw = Yraw.astype(float)
        Yc = gaussian_filter1d(Yraw.astype(float), lens/binsize*10, axis=-1, order=0, mode='nearest', truncate=5)
        assert Yc.shape == Yraw.shape
        Y = Yc[:, 5::10][None, ...] #downsample again
        print('convolved Y:', Y.shape, Yc.shape, Yraw.shape)
        assert Y.shape == Y_nc.shape

    else:
        Y = raster(labs, dat, bins, thresh, dt, shift, shiftsize, binshift)


    if behavior:
        #### bin behavior ####
        print('binning locations')
        locs = binned_statistic(ts, dat['cursor'].T, bins = bins, statistic = 'mean')[0].T #cursor position
        print(locs.shape)
        print('binning targets')
        targets = binned_statistic(ts, dat['target'].T, bins = bins, statistic = 'mean')[0].T #target position

        if shift and shiftsize > 0:
            locs = locs[binshift:]
            targets = targets[binshift:]
        elif shift and shiftsize < 0:
            locs = locs[:-binshift:]
            targets = targets[:-binshift]

    if subsamp:
        #### subselect data for initial analyses ####
        tex = np.arange(int(1000*100/binsize), int(2500*100/binsize)) #consider 150s of data for example analyses
        Y = Y[..., tex].astype(float)
        if behavior:
            locs = locs[tex, :] #subsample locs and targets
            targets = targets[tex, :]

    if behavior:
        return Y, locs, targets
    else:
        return Y
    

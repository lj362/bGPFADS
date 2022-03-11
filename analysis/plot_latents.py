import numpy as np
import time
import torch
import mgplvm as mgp
import pickle
import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils import detach, basedir, not_in
from load_joey import load_data
from scipy.stats import pearsonr, binned_statistic
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn import decomposition
import sys
device = mgp.utils.get_device()
#device = torch.device("cuda:0")
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
thresh = True

##### load data ######
region = 'M1'
fast_modes = True #only use latents with a timescale <= fast_thresh ms
fast_thresh = 200
binsize = 25
d_fit = 25
labstr = ''

savestr = '_thresh'

nfa = 20
fast_thresh = 200
    
for Lmax_shift in [-3]: #75ms prior to movement onset

    #### load data and compute reach angles ####
    Y, locs, targets = load_data(region = region, binsize = binsize, behavior = True, shift = False)

    #### load model ####
    fname = 'NegBinom_d'+str(d_fit)+'_'+region+'_b'+str(binsize)+'_circ'+labstr
    mod = torch.load('data/'+fname+'.pt').to(device)
    lats = detach(mod.lat_dist.lat_mu) #n_trials x m x d
    scale_args = np.argsort(-detach(mod.obs.dim_scale).flatten())
    lats = lats[0, ...][:, scale_args] #sort by information content
    lats_all = detach(mod.lat_dist.lat_mu)[0, ...][:, scale_args] #all latents
    ells = detach(mod.lat_dist.ell).flatten()[scale_args]
    if fast_modes: #only consider fast modes
        lats = lats[:, ells <= fast_thresh/binsize]

    ##### consider only the movement period ####
    #time points look like [0,   1,  2,   3,  4,  5,  6]
    #switch looks like     [t1, t1, t1, t12, t2, t2, t2]
    #deltas look like      [0,   0,  0,  d1, d2,  0,  0]
    #switches look like [3, 4]
    #dswitches look like [10, 1]
    #inds look like [0]
    #new switches look like [3]; index of the mixed bin!
    #pres and posts are [2], and [4] respectively
    deltas = np.abs(np.concatenate([np.zeros(1), (targets[1:, 0] - targets[:-1, 0])]))
    switches = np.where(deltas > 1e-5)[0] #change occurs during time bin s
    print(switches)
    dswitches = np.concatenate([np.ones(1)*10, switches[1:] - switches[:-1]])
    inds = np.zeros(len(switches)).astype(bool)
    inds[dswitches > 1.5] = 1
    switches = switches[inds] #only retain the index of the first switched bin
    print(switches)
    print('number of reaches:', len(switches))

    pres = np.array([targets[s-1, :] for s in switches])
    posts = np.array([targets[s+1, :] for s in switches])
    vecs = posts-pres
    mags = np.sqrt(np.sum(vecs**2, axis = 1, keepdims = True))
    nvecs = vecs / mags
    angs = np.sign(nvecs[:, 1]+1e-10) * np.arccos(nvecs[:, 0]) #map 0 to 1 in converting to polar coordinates


    print('lat shape:', lats.shape)

    #### run FA and extract velocity #####
    fa = decomposition.FactorAnalysis(nfa)
    Yfa = np.sqrt(Y[0, ...].T)
    Yfa = fa.fit_transform( Yfa - np.mean(Yfa, axis = 0, keepdims = True) ).T #nfa x T
    ts = np.arange(locs.shape[0])
    vels = np.concatenate([np.zeros( (1, 2) ), (locs[1:, :] - locs[:-1, :])], axis = 0)

    data_labels = ['bGPFA', 'Y', 'FA', 'vel', 'lats_all']
    datasets = [lats, Y[0, ...].T, Yfa.T, vels, lats_all] # (T x dim)

    ##### compute latents for all reaches ######
    all_inds = [] #indices (list of arrays)
    all_locs = [] #hand positions (list of arrays)
    all_targets = [] #target positions (list of arrays)
    all_data = [[], [], [], [], []] #corresponding data (4 lists of arrays)
    all_data_init = [[], [], [], [], []]
    all_data_move = [[], [], [], [], []]
    all_Lmaxs = []
    all_Ls = []
    all_RTs = []
    all_errs = []
    all_angs = []

    all_target_bins = []
    all_move_bins = []
    all_target_ends = []
    
    RT_hist = []
    for r, ang in enumerate(angs): #consider each reach
        if r == (len(angs)-1): #last reach
            s1, s2 = switches[r], lats.shape[0]
        else:
            s1, s2 = switches[r], switches[r+1] #mixed bin for both

        ds = s2-1 - (s1+1)
        dtarget = vecs[r]
        dcursor = locs[s2-1, :] - locs[s1+1, :]
        err = np.sqrt(np.sum((dtarget - dcursor)**2)) #difference between target and actual reach in mm
        all_errs.append(err)

        if ds > (100/binsize) and err <= 8: # at least 100ms and within 8mm of target
            #from first clean bin (s1+1) to last clean bin (s2-1) for constant target

            ####store movement at target and movement onset #####
            newlocs = locs[s1+1:s2, :]
            treach = newlocs.shape[0]
            ts = np.arange(treach) #measured in bins
            cs = CubicSpline(ts, newlocs) #cubic splines
            vs = cs(ts, 1) #first derivative

            vs = vels[s1+1:s2, :]
            reach_speeds = np.sqrt(np.sum(vs**2, axis = -1))
            try:
                Lmax = np.where(reach_speeds > 0.025*25)[0][0]-1
            except IndexError: #never exceeds threshold
                Lmax = 0
            RT = Lmax*binsize
            Lmax = Lmax + Lmax_shift
            
            RT_hist.append(RT)
            if (not thresh) or (Lmax - Lmax_shift) >=4: #require at least 100ms RT to include

                all_locs.append(locs[s1+1:s2, :]) # (Treach x 6)
                all_targets.append(targets[s1+1:s2, :]) # (Treach x 2)
                all_inds.append(np.arange(s1+1, s2)) # (Treach)

                all_target_bins.append(s1)
                all_target_ends.append(s2-1)

                all_angs.append(ang)
                for idata in range(5):
                    all_data[idata].append(datasets[idata][s1+1:s2, :])

                Lmax = max(0, Lmax) #can't be smaller than zero
                all_Lmaxs.append(Lmax)
                all_RTs.append(RT)
                all_Ls.append(treach)
                for idata in range(5):
                    all_data_init[idata].append(all_data[idata][-1][0, :])
                    all_data_move[idata].append(all_data[idata][-1][Lmax, :])

                all_move_bins.append(Lmax - Lmax_shift)


    all_data_init = [np.array(all_data_init[idata]) for idata in range(5)] #nreach x D
    all_data_move = [np.array(all_data_move[idata]) for idata in range(5)]
    all_RTs = np.array(all_RTs)
    angs = np.array(all_angs)
    all_Ls = np.array(all_Ls)
    all_Lmaxs = np.array(all_Lmaxs)
    print('reaches:', len(all_RTs), ' zero Lmaxs:', np.sum(np.array(all_Lmaxs) == 0))

    all_target_bins = np.array(all_target_bins)
    all_move_bins = np.array(all_move_bins)
    all_target_ends = np.array(all_target_ends)


    pickle.dump([all_data[0], angs, all_Lmaxs, all_RTs], open('../figures/figure_data/primate/'+fname+'_latents_angs'+savestr+'.pickled', 'wb'))
    pickle.dump([all_data[3], all_move_bins], open('../figures/figure_data/primate/'+fname+'_vels'+savestr+'.pickled', 'wb'))

    def dist(arr, args = None):
        '''arr is nreach x D/N'''
        if args is not None:
            arr = arr[args, :]
        ds = arr[:, None, :] - arr[None, :, :] #nreach x nreach x D
        ds = np.sqrt(np.sum(ds**2, axis = -1))
        nreach = ds.shape[0]
        np.fill_diagonal(ds, np.nan)
        return ds

    args = np.argsort(angs)
    print(args.shape)
    init_sims, move_sims = [], []
    for idata in range(4):
        init_sims.append(dist(all_data_init[idata], args))
        move_sims.append(dist(all_data_move[idata], args))
    dangs = angs[args][:, None] - angs[args][None, :]


    print('plotting pairwise distances')
    labs = [lab+'_init' for lab in data_labels[:4]] + [lab+'_move' for lab in data_labels[:4]]
    fig, axs = plt.subplots(2,4, figsize = (24, 12))
    naxs = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    for i, sims in enumerate(init_sims+move_sims):
        ax = axs[naxs[i][0], naxs[i][1]]
        sims = -sims
        ax.imshow(sims, cmap = 'coolwarm', vmin = np.nanquantile(sims, 0.01), vmax = np.nanquantile(sims, 0.99))
        ax.set_title(labs[i])
        pickle.dump(sims, open('../figures/figure_data/primate/'+fname+'_similarity_mat_'+labs[i]+savestr+'.pickled', 'wb'))

    plt.savefig('figures/sims.png', bbox_inches = 'tight')
    plt.close()

    print('computing binned statistics')
    flat_angs = (dangs.flatten() + np.pi) % (2*np.pi) - np.pi
    fig, axs = plt.subplots(2,4, figsize = (24, 12))
    bins = np.linspace(-np.pi, np.pi, 16)
    xs = (bins[1:] + bins[:-1]) / 2
    modulations = []
    for i, sims in enumerate(init_sims+move_sims):
        flat_sims = sims.flatten()
        flat_sims = (flat_sims - np.nanmean(flat_sims)) / np.nanstd(flat_sims) #compute z score for comparable scale
        means, _, _ = binned_statistic(flat_angs, flat_sims, statistic = np.nanmean, bins = bins)
        stds, _, _ = binned_statistic(flat_angs, flat_sims, statistic = np.nanstd, bins = bins)
        sems = stds / np.sqrt(np.histogram(flat_angs[~np.isnan(flat_sims)], bins = bins)[0])
        ax = axs[naxs[i][0], naxs[i][1]]
        ax.plot(xs, means, 'k-')
        ax.fill_between(xs, means-sems, means+sems, color = 'k', alpha = 0.2)
        ax.set_title(labs[i])
        modulations.append(np.amax(means) - np.amin(means)) #max minus min
        pickle.dump([xs, means, sems], open('../figures/figure_data/primate/'+fname+'_avg_similarity_'+labs[i]+savestr+'.pickled', 'wb'))
    plt.savefig('figures/sim_angs.png', bbox_inches = 'tight')
    plt.close()
    print('modulations:', modulations)
    pickle.dump([labs, modulations], open('../figures/figure_data/primate/'+fname+'_avg_similarity_modulation_'+str(Lmax_shift)+savestr+'.pickled', 'wb'))


    ####consider correlation with RT#####

    bins = np.linspace(-np.pi, np.pi, 21) #binning for computing prep states
    nbins = len(bins)-1
    min_RT, max_RT = 125, 425

    data_diffs = [[], [], [], [], []]
    RTs_corr = []
    lat_dists = np.zeros(len(angs))
    for ibin in range(nbins):
        inds = np.where( (angs <= bins[ibin+1]) & (angs > bins[ibin]) )[0]
        print(np.round(bins[ibin],2), np.round(bins[ibin+1],2), len(inds), 'reaches')
        RTs_corr.append(all_RTs[inds])
        for idata in range(5):
            avg_move = np.mean(all_data_move[idata][inds, :], axis = 0, keepdims = True) #target latent state
            diffs = np.sqrt(np.sum((all_data_init[idata][inds, :] - avg_move)**2, axis = 1))
            data_diffs[idata].append(diffs)
            if idata == 0:
                lat_dists[inds] = diffs #latent distance in original ordering
    RTs_corr = np.concatenate(RTs_corr)
    RT_args = np.where( (RTs_corr <= max_RT) & (RTs_corr >= min_RT) )[0]# & (all_inds < 400))[0]
    print(len(RT_args), 'of', len(RTs_corr))

    rvals = []
    for idata in [0, 1, 2, 4]:
        diffs = np.concatenate(data_diffs[idata])
        rvals.append(pearsonr(diffs[RT_args], RTs_corr[RT_args])[0])
        print(data_labels[idata], pearsonr(diffs[RT_args], RTs_corr[RT_args]))

        if idata == 0: #for bGPFA
            plt.figure()
            plt.scatter(diffs[RT_args], RTs_corr[RT_args], c = 'k', marker = 'o', alpha = 0.5, s = 5)
            plt.savefig('figures/RT_corrs.png', bbox_inches = 'tight')
            plt.close()

            data = [diffs[RT_args], RTs_corr[RT_args], RTs_corr]
            pickle.dump(data, open('../figures/figure_data/primate/'+fname+'_RTs_dists'+savestr+'.pickled', 'wb'))

    plt.figure()
    plt.hist(RTs_corr, color = 'k', bins = np.arange(0, 2000, binsize))
    plt.savefig('figures/RT_hist.png', bbox_inches = 'tight')
    plt.close()

    print('ells:', detach(mod.lat_dist.ell).flatten()[scale_args]*binsize)

    synthetic_RT = True
    nreps = 50000
    if synthetic_RT: #generate synthetic control
        print('generating synthetic data')
        ells = detach(mod.lat_dist.ell).flatten()
        if fast_modes:
            ells = ells[ells <= fast_thresh/binsize]
        print(ells)
        D = len(ells)

        #### generate synthetic data ####
        Lmaxs, nreach = np.array(all_Lmaxs), len(all_Lmaxs)
        buf = int(np.ceil(3*np.amax(ells)))
        tmax = int(np.round(max_RT / binsize)) + 2*buf

        #### using K and cholesky ####
        Lmaxs = np.maximum(Lmaxs, 1) #longer reaches will be discarded anyways
        ts = np.concatenate([np.zeros((nreach, 1)), Lmaxs.reshape(nreach, 1)], axis = 1) # (nreach x 2)
        print(ts)
        dts_sq = (ts[..., None] - ts[:, None, :])[:, None, ...]**2 # nreach x 1 x 2 x 2
        print('dts_sq:', dts_sq.shape)
        ells = ells[None, :, None, None] # 1 x D x 1 x 1
        K = np.exp(-dts_sq / (2 * ells**2)) # (nreach x D x 2 x 2)
        L = np.linalg.cholesky(K) # (nreach x D x 2 x 2)
        vs = np.random.normal(0, 1, (nreach,  D, 2, nreps))
        syn_lats = L @ vs # nreach x D x 2 x nreps
        syn_init_lats = syn_lats[:, :, 0, :] # nreach x D x nreps
        syn_move_lats = syn_lats[:, :, 1, :]
        print('syn_move:', syn_move_lats.shape)

        data_diffs, RTs_corr = [], []
        for ibin in range(nbins):
            inds = np.where( (angs <= bins[ibin+1]) & (angs > bins[ibin]) )[0] #(ndir, )
            RTs_corr.append(all_RTs[inds])
            avg_move = np.mean(syn_move_lats[inds, ...], axis = 0, keepdims = True) #target latent state (1 x D x nreps)
            diffs = np.sqrt(np.sum((syn_init_lats[inds, ...] - avg_move)**2, axis = 1)) #deltas (ndir x nreps)
            print(diffs.shape)
            data_diffs.append(diffs)
        RTs_corr = np.concatenate(RTs_corr)
        RT_args = np.where( (RTs_corr <= max_RT) & (RTs_corr >= min_RT) )[0]# & (all_inds < 400))[0]
        diffs = np.concatenate(data_diffs, axis = 0) #nreach x nreps
        print(diffs.shape)

        rs_syn = []
        for nrep in range(nreps):
            rs_syn.append(pearsonr(diffs[RT_args, nrep], RTs_corr[RT_args])[0])
            if nrep % 1000 == 0:
                print(nrep, rs_syn[-1])
        print('synthetic:', np.mean(rs_syn), np.std(rs_syn))

        print('writing synthetic RT')
        pickle.dump(rs_syn, open('../figures/figure_data/primate/'+fname+'_RTs_synthetic'+savestr+'.pickled', 'wb'))




    ### check if any latents correlate with RT directly "motivation states" ####
    lat_ells = detach(mod.lat_dist.ell).flatten()[scale_args]
    lat_scales = detach(mod.obs.dim_scale).flatten()[scale_args]

    RT_args_tot = np.where( (all_RTs <= max_RT) & (all_RTs >= min_RT) )[0]
    lat_rs = []
    for idata in range(1, 5):
        print('\n')
        dat = all_data_init[idata]
        for idim in range(dat.shape[1]):
            if idata == 4: #print timescale, prior scale, and correlation with principal dimension
                print(data_labels[idata], idim, lat_ells[idim]*binsize, np.log(lat_scales[idim]), pearsonr(dat[RT_args_tot, idim], all_RTs[RT_args_tot]))#, pearsonr(dat[:, 0], dat[:, idim]))
                lat_rs.append(pearsonr(dat[RT_args_tot, idim], all_RTs[RT_args_tot])[0])
            elif idata in [2, 3] or idim % 10 == 0:
                print(data_labels[idata], idim, pearsonr(dat[RT_args_tot, idim], all_RTs[RT_args_tot]))

    Yz = (Y - np.mean(Y, axis = (0, 2), keepdims = True)) / np.std(Y, axis = (0, 2), keepdims = True)
    print('mean z scored activity', pearsonr(np.mean(Yz, axis = (0, 1))[RT_args_tot], all_RTs[RT_args_tot]))
    print('mean activity:', pearsonr(np.mean(Y, axis = (0, 1))[RT_args_tot], all_RTs[RT_args_tot]))

    print(Y.shape, lats_all.shape)
    print('mode 0 correlation with mean activity', pearsonr(lats_all[:, 0], np.mean(Y, axis = (0, 1))))
    print('fa 0 correlation with mean activity', pearsonr(np.mean(Yfa, axis = 0), np.mean(Y, axis = (0, 1))))
    print('mode 0 correlation with fa 0', pearsonr(lats_all[:, 0], np.mean(Yfa, axis = 0)))

    plt.figure()
    plt.hist(all_Ls[RT_args_tot]*binsize, color = 'k', bins = np.arange(0, 4000, 100))
    plt.savefig('figures/trial_length_hist.png', bbox_inches = 'tight')
    plt.close()

    plt.figure()
    plt.scatter(dat[RT_args_tot, np.argmax(np.array(lat_rs)**2)], all_RTs[RT_args_tot], c = 'k')
    plt.savefig('figures/max_lat_corr.png', bbox_inches = 'tight')
    plt.close()

    data = [dat[:, np.argmax(np.array(lat_rs)**2)], all_RTs, all_Ls*binsize]
    pickle.dump(data, open('../figures/figure_data/primate/'+fname+'_RTs_lats'+savestr+'.pickled', 'wb'))
    pickle.dump(RT_hist, open('../figures/figure_data/primate/'+fname+'_RT_hist'+savestr+'.pickled', 'wb'))

    def comp_r2(X, Y):
        regressor = LinearRegression(fit_intercept=True)
        regressor.fit(X, Y)
        return regressor.score(X, Y)

    long_lats = all_data_init[4][:, lat_ells >= 1500/binsize] #long timescale latents
    long_ells = np.concatenate([lat_ells[lat_ells >= 1500/binsize], np.ones(1)*np.nan])
    lat_dists = lat_dists[:, None]
    print(long_lats.shape, lat_dists.shape)
    X = np.concatenate([long_lats, lat_dists], axis = 1)[RT_args_tot, :] #T x D'+1
    Y = all_RTs[RT_args_tot, None] #T x 1
    print('all:', comp_r2(X, Y))
    for i in range(X.shape[1]):
        print(i, 'r2', comp_r2(X[:, i:i+1], Y), 'ell', long_ells[i]*binsize)

    X = np.concatenate([long_lats[:, np.argmax(long_ells[:-1])][:, None], lat_dists], axis = 1)[RT_args_tot, :] #T x 2
    print('comb r2', comp_r2(X, Y))

    print('correlation with time:', pearsonr(RT_args_tot, all_RTs[RT_args_tot])[0])

    X = np.concatenate([long_lats[RT_args_tot, 1:2], RT_args_tot[:, None]], axis = 1)
    print('time:', comp_r2(X[:, 1:], Y))
    print('long:', comp_r2(X[:, :1], Y))
    print('long + time:', comp_r2(X, Y))

    print(Lmax_shift)
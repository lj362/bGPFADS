from pyparsing import TokenConverter
import torch
import numpy as np
from torch import clamp, nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from ..utils import softplus, inv_softplus
from ..manifolds.base import Manifold
from .common import Rdist
from typing import Optional
from ..fast_utils.toeplitz import sym_toeplitz_matmul


class GPbase(Rdist):
    name = "GPbase"  # it is important that child classes have "GP" in their name, this is used in control flow

    def __init__(self,
                 manif: Manifold,
                 m: int,
                 n_samples: int,
                 ts: torch.Tensor,
                 _scale=0.9,
                 ell=None,
                 _alpha = 1.0,
                 ):
        """
        Parameters
        ----------
        manif: Manifold
            manifold of ReLie
        m : int
            number of conditions/timepoints
        n_samples: int
            number of samples
        ts: Tensor
            input timepoints for each sample (n_samples x 1 x m)
        mu : Optional[np.ndarray]
            initialization of the vartiational means (m x d2)
            
        Notes
        -----
        Our GP has prior N(0, K)
        We parameterize our posterior as N(K2 v, K2 I^2 K2)
        where K2 K2 = K and I(s) is some inner matrix which can take different forms.
        s is a vector of scale parameters for each time point.
        
        """

        super(GPbase, self).__init__(manif, 1)  #kmax = 1

        self.manif = manif
        self.d = manif.d
        self.m = m

        #initialize GP mean parameters
        nu = torch.randn((n_samples, self.d, m)) * 0.01
        self._nu = nn.Parameter(data=nu, requires_grad=True)  #m in the notes

        _alpha = torch.clamp(torch.tensor(1.0),0,1)
        self._alpha = nn.Parameter(_alpha, requires_grad=True)

        nu2 = torch.randn((n_samples, self.d, m)) * 0.01
        self._nu2 = nn.Parameter(data=nu2, requires_grad=True)  #m in the notes
        #print('nu dim',nu.shape)

        #initialize covariance parameters
        _scale = torch.ones(n_samples, self.d, m) * _scale  #n_diag x T
        self._scale = nn.Parameter(data=inv_softplus(_scale),
                                   requires_grad=True)

        #initialize length scale
        ell = (torch.max(ts) - torch.min(ts)) / 20 if ell is None else ell
        _ell = torch.ones(1, self.d, 1) * ell
        self._ell = nn.Parameter(data=inv_softplus(_ell), requires_grad=True)

        #pre-compute time differences (only need one row for the toeplitz stuff)
        self.ts = ts
        dts_sq = torch.square(ts - ts[..., :1])  #(n_samples x 1 x m)
    
        #sum over _input_ dimension, add an axis for _output_ dimension
        dts_sq = dts_sq.sum(-2)[:, None, ...]  #(n_samples x 1 x m)
        self.dts_sq = nn.Parameter(data=dts_sq, requires_grad=False)

        dts_2 = ts - ts[..., :1]
        dts_2 = dts_2.sum(-2)[:, None, ...]
        self.dts_2 = nn.Parameter(data=dts_2, requires_grad=False)

        dts_first = torch.square(ts[..., :1])  #(n_samples x 1 x m)
        #sum over _input_ dimension, add an axis for _output_ dimension
        dts_first = dts_first.sum(-2)[:, None, ...]  #(n_samples x 1 x m)
        self.dts_first = nn.Parameter(data=dts_first, requires_grad=False)

        dts_last = torch.square(ts[..., -1::])  #(n_samples x 1 x m)
        #sum over _input_ dimension, add an axis for _output_ dimension
        dts_last = dts_last.sum(-2)[:, None, ...]  #(n_samples x 1 x m)
        self.dts_last = nn.Parameter(data=dts_last, requires_grad=False)


        self.dt = (ts[0, 0, 1] - ts[0, 0, 0]).item()  #scale by dt

    @property
    def scale(self) -> torch.Tensor:
        return softplus(self._scale)

    @property
    def nu(self) -> torch.Tensor:
        return self._nu

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha

    @property
    def nu2(self) -> torch.Tensor:
        return self._nu2

    @property
    def ell(self) -> torch.Tensor:
        return softplus(self._ell)

    @property
    def prms(self):
        return self.nu, self.scale, self.ell, self.nu2, self.alpha

    @property
    def lat_mu(self):
        """return variational mean mu = K_half @ nu"""
        nu = self.nu
        nu2 = self.nu2
        K_half = self.K_half()  #(n_samples x d x m)
        K_half_tl = self.K_half_antidiag()
        mu = sym_toeplitz_matmul(K_half, nu[..., None])[..., 0]
        mu2 = sym_toeplitz_matmul(K_half_tl,nu2[..., None])[..., 0]
        mu = mu + mu2
        return mu.transpose(-1, -2)  #(n_samples x m x d)


    def K_half(self, sample_idxs=None):
        """compute one column of the square root of the prior matrix"""
        '''compute the blocks on the main diagonal'''
        nu = self.nu  #mean parameters

        #K^(1/2) has length scale ell/sqrt(2) if K has ell
        ell_half = self.ell / np.sqrt(2)

        #K^(1/2) has sig var sig*2^1/4*pi^(-1/4)*ell^(-1/2) if K has sig^2 (1 x d x 1)
        sig_sqr_half = 1 * (2**(1 / 4)) * np.pi**(-1 / 4) * self.ell**(
            -1 / 2) * self.dt**(1 / 2)

        if sample_idxs is None:
            dts = self.dts_sq[:, ...]
        else:
            dts = self.dts_sq[sample_idxs, ...]

        if sample_idxs is None:
            dts2 = self.dts_2[:, ...]
        else:
            dts2 = self.dts_2[sample_idxs, ...]

        if sample_idxs is None:
            dts_first = self.dts_first[:, ...]
        else:
            dts_first = self.dts_first[sample_idxs, ...]  

        if sample_idxs is None:
            dts_last = self.dts_last[:, ...]
        else:
            dts_last = self.dts_last[sample_idxs, ...]          

        # (n_samples x d x m)
        #K_half = sig_sqr_half * torch.exp(-dts / (2 * torch.square(ell_half)))
        
        tau = dts2 / ell_half
        tau2 = dts / torch.square(ell_half)
        K_half_tr = tau / (1 + tau2)
        K_half_bl = - tau / (1 + tau2)
        K_half_tl = 1 / (1 + tau2)

        #print('dts:',dts.shape,type(dts))
        #print('dts2:',dts2.shape,type(dts2))

        std1 = torch.std(dts2[0,0,:].to(torch.float64))# in this case not needed
        std2 = torch.std(dts2[0,0,:].to(torch.float64))# in this case not needed
        var1 = torch.var(dts2[0,0,:].to(torch.float64))
        var2 = torch.var(dts2[0,0,:].to(torch.float64))

        K_half_tl[...,0,:] = K_half_tl[...,0,:] #* var1 * alpha
        K_half_tl[...,1,:] = K_half_tl[...,1,:] #* var2 * alpha
        K_half_br = 1 / (1 + tau2) 
            
        #K_half = torch.cat((K_half_tl,K_half_bl),2)
        
        K_half = K_half_tl
        #print('K_half(diagonal blocks) dim = ',K_half.shape)
        return K_half


    def I_v(self, v, sample_idxs=None):
        """
        Compute I @ v for some vector v.
        This should be implemented for each class separately.
        v is (n_samples x d x m x n_mc) where n_samples is the number of sample_idxs
        """
        pass

    def kl(self, batch_idxs=None, sample_idxs=None):
        """
        Compute KL divergence between prior and posterior.
        This should be implemented for each class separately
        """
        pass

    def K_half_antidiag(self, sample_idxs=None):
        """compute one column of the square root of the prior matrix"""
        '''compute the blocks on the main diagonal'''
        nu = self.nu  #mean parameters
        alpha = self.alpha

        #K^(1/2) has length scale ell/sqrt(2) if K has ell
        ell_half = self.ell / np.sqrt(2)

        #K^(1/2) has sig var sig*2^1/4*pi^(-1/4)*ell^(-1/2) if K has sig^2 (1 x d x 1)
        sig_sqr_half = 1 * (2**(1 / 4)) * np.pi**(-1 / 4) * self.ell**(
            -1 / 2) * self.dt**(1 / 2)

        if sample_idxs is None:
            dts = self.dts_sq[:, ...]
        else:
            dts = self.dts_sq[sample_idxs, ...]

        if sample_idxs is None:
            dts2 = self.dts_2[:, ...]
        else:
            dts2 = self.dts_2[sample_idxs, ...]

        if sample_idxs is None:
            dts_first = self.dts_first[:, ...]
        else:
            dts_first = self.dts_first[sample_idxs, ...]  

        if sample_idxs is None:
            dts_last = self.dts_last[:, ...]
        else:
            dts_last = self.dts_last[sample_idxs, ...]          

        # (n_samples x d x m)
        
        tau = dts2 / ell_half
        tau2 = dts / torch.square(ell_half)
        K_half_tr = alpha * tau / (1 + tau2)
        K_half_bl = - tau / (1 + tau2)
        K_half_tl = 1 / (1 + tau2)
        K_half_br = 1 / (1 + tau2) 
        #print('K_half_tr dim = ',K_half_tr.shape,'K_half dim = ',K_half.shape)
        #K_half_tr[...,-1,:] = (-1) * K_half_tr[...,-1,:]


        std1 = torch.std(dts2[0,0,:].to(torch.float64))
        std2 = torch.std(dts2[0,0,:].to(torch.float64))
        var1 = torch.var(dts2[0,0,:].to(torch.float64))# in this case not needed
        var2 = torch.var(dts2[0,0,:].to(torch.float64))# in this case not needed

        
        K_half_tr[...,0,:] = K_half_tr[...,0,:] #* std1 * std2 * alpha
        K_half_tr[...,1,:] = K_half_tr[...,1,:] * (-1)#* std1 * std2 * (-1) * alpha
        return K_half_tr



        """
        Compute KL divergence between prior and posterior.
        This should be implemented for each class separately
        """
        pass

    def full_cov(self):
        """Compute the full covariance Khalf @ I @ I @ Khalf"""
        v = torch.diag_embed(torch.ones(
            self._scale.shape))  #(n_samples x d x m x m)
        I = self.I_v(v)  #(n_samples x d x m x m)
        K_half = self.K_half()  #(n_samples x d x m)

        Khalf_I = sym_toeplitz_matmul(K_half, I)  #(n_samples x d x m x m)
        K_post = Khalf_I @ Khalf_I.transpose(-1, -2)  #Kpost = Khalf@I@I@Khalf
        #print('K_post dim = ',K_post.shape)
        return K_post.detach()

    def sample(self,
               size,
               Y=None,
               batch_idxs=None,
               sample_idxs=None,
               kmax=5,
               analytic_kl=False,
               prior=None):
        """
        generate samples and computes its log entropy
        """

        #compute KL analytically
        lq = self.kl(batch_idxs=batch_idxs,
                     sample_idxs=sample_idxs)  #(n_samples x d)

        K_half = self.K_half(sample_idxs=sample_idxs)  #(n_samples x d x m)
        K_half_tr = self.K_half_antidiag(sample_idxs=sample_idxs)
        n_samples, d, m = K_half.shape

        # sample a batch with dims: (n_samples x d x m x n_mc)
        v = torch.randn(n_samples, d, m, size[0])  # v ~ N(0, 1)
        v2 = torch.randn(n_samples, d, m, size[0])  # v2 ~ N(0, 1) for antidiagonal blocks
        #compute I @ v (n_samples x d x m x n_mc)
        I_v = self.I_v(v, sample_idxs=sample_idxs)
        I_v_2 = self.I_v(v2, sample_idxs=sample_idxs)

        nu = self.nu  #mean parameter (n_samples, d, m)
        nu2 = self.nu2
        alpha = self.alpha
        if sample_idxs is not None:
            nu = nu[sample_idxs, ...]
        samp = nu[..., None] + I_v  #add mean parameter to each sample

        if sample_idxs is not None:
            nu2 = nu2[sample_idxs, ...]
        samp2 = nu2[..., None] + I_v_2  #add mean parameter to each sample

        #compute K@(I@v+nu)
        x = sym_toeplitz_matmul(K_half, samp)  #(n_samples x d x m x n_mc)
        x = x.permute(-1, 0, 2, 1)  #(n_mc x n_samples x m x d)

        if batch_idxs is not None:  #only select some time points
            x = x[..., batch_idxs, :]
        #print('x has shape of ',x.shape)
        #(n_mc x n_samples x m x d), (n_samples x d)

        x_2 = sym_toeplitz_matmul(K_half_tr, samp2)  #(n_samples x d x m x n_mc)
        x_2 = x_2.permute(-1, 0, 2, 1)  #(n_mc x n_samples x m x d)
        #print('x_2 dim =', x_2.shape,',x dim')

        if batch_idxs is not None:  #only select some time points
            x_2 = x_2[..., batch_idxs, :]
        #print('x has shape of ',x.shape)
        #(n_mc x n_samples x m x d), (n_samples x d)

        x = x + x_2
        return x, lq

    def gmu_parameters(self):
        return [self.nu]

    def concentration_parameters(self):
        return [self._scale, self._ell]

    def msg(self, Y=None, batch_idxs=None, sample_idxs=None):

        mu_mag = torch.sqrt(torch.mean(self.nu**2)).item()
        sig = torch.median(self.scale).item()
        ell = self.ell.mean().item()

        string = (' |mu| {:.3f} | sig {:.3f} | prior_ell {:.3f} |').format(
            mu_mag, sig, ell)

        return string

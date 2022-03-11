import numpy as np
from mgplvm import quaternion
from scipy import special
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .base import Manifold
from typing import Tuple, Optional, List
from ..inducing_variables import InducingPoints


class S3(Manifold):

    def __init__(self,
                 m: int,
                 d: Optional[int] = None,
                 mu: Optional[np.ndarray] = None,
                 Tinds: Optional[np.ndarray] = None,
                 initialization: Optional[str] = 'identity',
                 Y: Optional[np.ndarray] = None):
        super().__init__(d=d)

        self.m = m
        self.d2 = d + 1  # dimensionality of the group parameterization

        mudata = self.initialize(initialization, m, d, Y)
        if mu is not None:
            mudata[Tinds, ...] = torch.tensor(mu,
                                              dtype=torch.get_default_dtype())

        self.mu = nn.Parameter(data=mudata, requires_grad=True)

        # per condition (V(Sd) = 2 pi^((d+1)/2) / Gamma((d+1)/2)
        self.lprior_const = torch.tensor(
            special.loggamma((d + 1) / 2) - np.log(2) -
            (d + 1) / 2 * np.log(np.pi))

    @staticmethod
    def initialize(initialization, m, d, Y):
        '''initializes latents - can add more exciting initializations as well'''
        # initialize at identity
        mudata = torch.tensor(np.array(
            [[1] + [0 for j in range(d)] for i in range(m)]),
                              dtype=torch.get_default_dtype())
        return mudata

    def inducing_points(self, n, n_z, z=None):
        if z is None:
            z = torch.randn(n, self.d2, n_z)
            z = z / torch.norm(z, dim=1, keepdim=True)

        return InducingPoints(n,
                              self.d2,
                              n_z,
                              z=z,
                              parameterise=lambda x: self.expmap2(x, dim=-2))

    @property
    def prms(self) -> Tensor:
        mu = self.mu
        norms = torch.norm(mu, dim=1, keepdim=True)
        return mu / norms

    @property
    def name(self):
        return 'Sphere(' + str(self.d) + ')'

    def lprior(self, g):
        return self.lprior_const * torch.ones(g.shape[:2])

    def transform(self,
                  x: Tensor,
                  batch_idxs: Optional[List[int]] = None) -> Tensor:
        mu = self.prms
        if batch_idxs is not None:
            mu = mu[batch_idxs]
        return self.gmul(mu, x)  # group multiplication

    @staticmethod
    def expmap(x: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
        '''TODO: implement'''
        return

    @staticmethod
    def expmap2(x: Tensor, dim: int = -1) -> Tensor:
        '''TODO: implement'''
        return

    @staticmethod
    def logmap(q: Tensor, dim: int = -1) -> Tensor:
        '''TODO: implement'''
        return

    @staticmethod
    def inverse(q: Tensor) -> Tensor:
        '''TODO: implement'''
        return

    @staticmethod
    def gmul(x: Tensor, y: Tensor) -> Tensor:
        '''TODO: implement'''
        return

    @staticmethod
    def log_q(log_base_prob, x, d, kmax, dim=-1):
        '''TODO: implement'''
        return

    @staticmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        cosdist = (x[..., None] * y[..., None, :])
        cosdist = cosdist.sum(-3)
        return 2 * (1 - cosdist)
